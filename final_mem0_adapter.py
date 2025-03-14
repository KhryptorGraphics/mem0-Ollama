"""
Integration adapter for mem0ai with Ollama

This module provides a proper integration between mem0ai and Ollama,
using Qdrant as the vector database for memory storage.
"""

import logging
import time
import os
import json
import uuid
import random
import threading
import sys
from typing import Dict, List, Any, Optional
from datetime import datetime

# Configure logging
logger = logging.getLogger(__name__)

# Setup paths to ensure mem0ai can be imported
try:
    # Add mem0 repository to sys.path
    REPO_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "external", "mem0")
    if REPO_PATH not in sys.path:
        sys.path.insert(0, REPO_PATH)
        logger.info(f"Added mem0 repository path to sys.path: {REPO_PATH}")
    
    # Try to import mem0ai - direct from repository
    from mem0.memory.main import Memory
    from mem0.configs.base import MemoryConfig
    HAS_MEM0AI = True
    logger.info("Successfully imported mem0.memory.main.Memory from repository")
except ImportError as e:
    logger.warning(f"Failed to import from repository: {e}")
    # Fall back to pip-installed package
    try:
        from mem0ai import Memory
        HAS_MEM0AI = True
        logger.info("Successfully imported Memory from mem0ai package")
    except ImportError as e:
        HAS_MEM0AI = False
        logger.warning(f"mem0ai module not found. Using fallback memory implementation. Error: {e}")
except Exception as e:
    HAS_MEM0AI = False
    logger.error(f"Unexpected error importing mem0ai: {e}")
    logger.warning("Using fallback memory implementation due to import error.")

from config import (
    OLLAMA_HOST,
    DEFAULT_MODEL,
    MEM0_HOST,
    MAX_MEMORIES,
    MEMORY_DATABASE
)

# Fallback memory store when mem0ai is not available
class MemoryStore:
    """Simple JSON-based memory store that's used when mem0ai isn't available"""
    
    def __init__(self, db_file="memory_store.json"):
        """Initialize the memory store"""
        self.db_file = db_file
        self.memories = {}
        self.lock = threading.RLock()
        self._load()
    
    def _load(self):
        """Load memories from file"""
        try:
            if os.path.exists(self.db_file):
                with open(self.db_file, 'r') as f:
                    data = json.load(f)
                    self.memories = data
                    logger.info(f"Loaded {sum(len(memories) for memories in self.memories.values() if isinstance(memories, list))} memories from {self.db_file}")
        except Exception as e:
            logger.error(f"Error loading memories: {e}")
            self.memories = {}
    
    def _save(self):
        """Save memories to file"""
        try:
            with open(self.db_file, 'w') as f:
                json.dump(self.memories, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving memories: {e}")
    
    def add(self, text, user_id="default", metadata=None):
        """Add a memory"""
        with self.lock:
            if user_id not in self.memories:
                self.memories[user_id] = []
            
            # Generate a unique ID
            memory_id = str(uuid.uuid4())
            
            # Create memory object
            memory = {
                "id": memory_id,
                "memory": text,
                "metadata": metadata or {},
                "created_at": time.time(),
                "embedding": None  # We won't store actual embeddings in JSON
            }
            
            # Add memory
            self.memories[user_id].append(memory)
            
            # Save to disk
            self._save()
            
            return {"id": memory_id}
    
    def get_all(self, user_id="default"):
        """Get all memories for a user"""
        with self.lock:
            if user_id not in self.memories:
                return []
            
            # Get memories and sort by created_at (newest first)
            memories = self.memories[user_id]
            memories = sorted(memories, key=lambda m: m.get("created_at", 0), reverse=True)
            
            return memories
    
    def search(self, query, user_id="default", limit=10):
        """
        Simple text-based search since we don't store actual embeddings
        
        Returns:
            Dict with 'results' key containing list of memory objects with 'id', 'memory', 
            'metadata', and 'score' (similarity score)
        """
        with self.lock:
            if user_id not in self.memories:
                return {"results": []}
            
            results = []
            query_lower = query.lower()
            
            for memory in self.memories[user_id]:
                # Simple text search
                memory_text = memory.get("memory", "").lower()
                
                # Calculate a simple score based on substring match
                if query_lower in memory_text:
                    score = 0.8  # High score for exact match
                else:
                    # Count word matches
                    query_words = set(query_lower.split())
                    memory_words = set(memory_text.split())
                    
                    common_words = query_words.intersection(memory_words)
                    
                    if common_words:
                        score = 0.5 * len(common_words) / len(query_words)
                    else:
                        score = 0.1  # Low score default
                
                results.append({
                    "id": memory.get("id"),
                    "memory": memory.get("memory"),
                    "metadata": memory.get("metadata", {}),
                    "score": score
                })
            
            # Sort by score (highest first) and apply limit
            results = sorted(results, key=lambda r: r.get("score", 0), reverse=True)
            results = results[:limit]
            
            return {"results": results}

class Mem0Adapter:
    """
    Adapter for mem0ai library that integrates with Ollama for embeddings
    and LLM capabilities, using Qdrant as the vector database for memory storage.
    """
    
    # Memory system modes
    MEMORY_MODE_AUTO = "auto"
    MEMORY_MODE_MEM0AI = "mem0ai"
    MEMORY_MODE_FALLBACK = "fallback"
    
    def __init__(self, 
                 ollama_host: str = OLLAMA_HOST, 
                 mem0_host: str = MEM0_HOST, 
                 default_model: str = DEFAULT_MODEL,
                 memory_mode: str = MEMORY_MODE_AUTO):
        """
        Initialize the mem0ai adapter with Ollama settings.
        
        Args:
            ollama_host: URL of the Ollama host
            mem0_host: URL of the Qdrant vector db
            default_model: Default Ollama model to use
            memory_mode: Force specific memory system (auto, mem0ai, fallback)
        """
        self.ollama_host = ollama_host
        self.mem0_host = mem0_host
        self.default_model = default_model
        self.memory_mode = memory_mode
        self.db_file = MEMORY_DATABASE  # Store db_file for fallback
        self._memory = None
        self._active_memories = {}
        self._initialized = False
        self.current_mode = None  # Which mode is currently active
        
        # Initialize the memory system
        self._initialize()
        
    def _initialize(self) -> None:
        """Initialize memory system with proper configuration based on memory_mode setting."""
        if self._initialized:
            return
            
        # Force fallback mode if specified
        if self.memory_mode == self.MEMORY_MODE_FALLBACK:
            logger.info("Using fallback memory store due to memory_mode setting")
            try:
                self._memory = MemoryStore(db_file=self.db_file)
                self._initialized = True
                self.current_mode = self.MEMORY_MODE_FALLBACK
                logger.info("Successfully initialized fallback memory store")
                return
            except Exception as e:
                logger.error(f"Failed to initialize fallback memory store: {e}")
                raise
            
        # Check if mem0ai is requested but not available
        if self.memory_mode == self.MEMORY_MODE_MEM0AI and not HAS_MEM0AI:
            logger.error("mem0ai was explicitly requested but is not available")
            raise ImportError("mem0ai module is not installed but was explicitly requested")
                
        # Check if mem0ai is available (auto mode) or explicitly requested
        if not HAS_MEM0AI and self.memory_mode == self.MEMORY_MODE_AUTO:
            logger.warning("mem0ai module not found, using fallback JSON-based memory store")
            # Use the provided db_file parameter if available
            try:
                self._memory = MemoryStore(db_file=self.db_file)
                self._initialized = True
                self.current_mode = self.MEMORY_MODE_FALLBACK
                logger.info("Successfully initialized fallback memory store")
                return
            except Exception as e:
                logger.error(f"Failed to initialize fallback memory store: {e}")
                raise
        
        # If we're here, either:
        # 1. mem0ai is available and mode is AUTO
        # 2. mem0ai is available and explicitly requested
        try:
            # Normalize host URL for better compatibility
            host = self.ollama_host
            if "localhost" not in host and "127.0.0.1" not in host:
                logger.warning(f"Ollama host is set to {host}, which may not be correct. Using http://localhost:11434 instead.")
                host = "http://localhost:11434"
            elif not host.startswith(('http://', 'https://')):
                host = f"http://{host}"
                
            logger.info(f"Initializing mem0ai with Ollama host: {host} and vector DB at {self.mem0_host}")
            
            # Get embedder model - make sure it's appropriate
            embedder_model = "nomic-embed-text"  # Default to a known embedding model
            
            try:
                # Try to detect a better embedding model
                import requests
                response = requests.get(f"{host}/api/tags", timeout=5)
                if response.status_code == 200:
                    models = [model.get("name") for model in response.json().get("models", [])]
                    
                    # Look for dedicated embedding models
                    if "nomic-embed-text" in models:
                        embedder_model = "nomic-embed-text"
                        logger.info(f"Using dedicated embedding model: {embedder_model}")
                    elif any("embed" in model.lower() for model in models):
                        # Find first model with 'embed' in the name
                        for model in models:
                            if "embed" in model.lower():
                                embedder_model = model
                                logger.info(f"Using dedicated embedding model: {embedder_model}")
                                break
                                
            except Exception as e:
                logger.warning(f"Error detecting embedding models: {e}")
                logger.warning(f"Using default embedding model: {embedder_model}")
                
            # Extract host and port from URLs
            ollama_host_name = host.replace("http://", "").replace("https://", "").split(":")[0]
            ollama_port = 11434  # Default Ollama port
            if ":" in host:
                try:
                    ollama_port = int(host.split(":")[-1].split("/")[0])
                except ValueError:
                    logger.warning(f"Could not parse port from {host}, using default port 11434")
            
            qdrant_host = self.mem0_host.replace("http://", "").replace("https://", "").split(":")[0]
            qdrant_port = 6333  # Default Qdrant port
            if ":" in self.mem0_host:
                try:
                    qdrant_port = int(self.mem0_host.split(":")[-1].split("/")[0])
                except ValueError:
                    logger.warning(f"Could not parse port from {self.mem0_host}, using default port 6333")
            
            # Reset the Qdrant collection first to avoid dimension mismatch errors
            # This happens because models generate different embedding dimensions
            try:
                import requests
                collection_name = "mem0_memories"
                
                # Delete collection if it exists
                delete_response = requests.delete(
                    f"{self.mem0_host}/collections/{collection_name}", 
                    timeout=5
                )
                
                logger.info(f"Deleted Qdrant collection: {delete_response.status_code}")
            except Exception as e:
                logger.warning(f"Error resetting Qdrant collection: {e}")
            
            # Configure mem0ai with parameters
            config = {
                "llm": {
                    "provider": "ollama",
                    "config": {
                        "model": self.default_model,
                        "ollama_base_url": host
                    }
                },
                "embedder": {
                    "provider": "ollama",
                    "config": {
                        "model": embedder_model,
                        "ollama_base_url": host,
                        # Explicitly set embedding dimensions to match the model
                        "embedding_dims": 4096  # This should match the model's output dimensions
                    }
                },
                "vector_store": {
                    "provider": "qdrant",
                    "config": {
                        "host": qdrant_host,
                        "port": qdrant_port,
                        "collection_name": "mem0_memories"
                    }
                },
                "version": "v1.1"  # Use the latest API version
            }
            
            logger.info(f"Using config: {config}")
            
            # Create the Memory instance
            try:
                # Following mem0ai's recommended initialization pattern
                self._memory = Memory.from_config(config)
                self._initialized = True
                self.current_mode = self.MEMORY_MODE_MEM0AI
                logger.info("Successfully initialized mem0ai Memory system")
                
                # Test the memory with a simple add/search to verify proper configuration
                test_id = self._memory.add(
                    "Test memory initialization", 
                    user_id="system"
                )
                
                if test_id and isinstance(test_id, dict) and "results" in test_id:
                    logger.info(f"Memory system test successful: {test_id}")
                else:
                    logger.warning("Memory system initialized but test add operation returned unexpected result")
                    
            except Exception as mem_error:
                logger.error(f"Error initializing mem0ai Memory: {mem_error}")
                
                # Don't use fallback if mem0ai was explicitly requested
                if self.memory_mode == self.MEMORY_MODE_MEM0AI:
                    logger.error("Not falling back because mem0ai was explicitly requested")
                    raise
                    
                # Try fallback if auto mode and mem0ai config fails
                logger.warning("Falling back to JSON-based memory store due to mem0ai initialization failure")
                self._memory = MemoryStore(db_file=self.db_file)
                self._initialized = True
                self.current_mode = self.MEMORY_MODE_FALLBACK
                logger.info("Successfully initialized fallback memory store")
                
        except Exception as e:
            logger.error(f"Failed to initialize memory system: {e}")
            
            # Don't use fallback if mem0ai was explicitly requested
            if self.memory_mode == self.MEMORY_MODE_MEM0AI:
                logger.error("Not using fallback because mem0ai was explicitly requested")
                raise
                
            # Final fallback for auto mode
            logger.warning("Using fallback memory store after all attempts failed")
            try:
                self._memory = MemoryStore(db_file=self.db_file)
                self._initialized = True
                self.current_mode = self.MEMORY_MODE_FALLBACK
            except:
                logger.error("Critical failure: Could not initialize any memory system")
                raise
            
    def add_memory(self, text: str, memory_id: str = "default", metadata: Optional[Dict[str, Any]] = None) -> Optional[str]:
        """
        Add a memory to the vector store with enhanced metadata.
        
        Args:
            text: Memory text to store
            memory_id: User ID for memory grouping
            metadata: Additional metadata
            
        Returns:
            Memory ID if successful, None otherwise
        """
        if not self._initialized:
            self._initialize()
            
        try:
            # Prepare enhanced metadata
            metadata = metadata or {}
            
            # Tag the source in metadata
            metadata["source"] = "mem0ai" if self.current_mode == self.MEMORY_MODE_MEM0AI else "fallback"
            
            # Add memory using mem0ai or fallback
            if self.current_mode == self.MEMORY_MODE_MEM0AI:
                # Use the updated mem0ai pattern which returns a dict with 'results'
                result = self._memory.add(
                    text, 
                    user_id=memory_id,
                    metadata=metadata
                )
                
                # Extract the memory ID from the result according to the API version
                if isinstance(result, dict) and "results" in result:
                    # New format: Get ID from first result
                    memories = result.get("results", [])
                    memory_id = memories[0].get("id") if memories else None
                else:
                    # Legacy format or direct result
                    memory_id = result.get("id") if isinstance(result, dict) else None
            else:
                # Fallback system returns directly
                result = self._memory.add(
                    text, 
                    user_id=memory_id,
                    metadata=metadata
                )
                memory_id = result.get("id")
                
            logger.info(f"Added memory with ID: {memory_id}")
            return memory_id
        except Exception as e:
            logger.error(f"Error adding memory: {e}")
            
            # If we got a vector dimension error, fall back to the simpler system
            if "vector dimension error" in str(e).lower() and self.current_mode == self.MEMORY_MODE_MEM0AI:
                logger.warning("Vector dimension mismatch. Falling back to the simpler store.")
                self.current_mode = self.MEMORY_MODE_FALLBACK
                self._memory = MemoryStore(db_file=self.db_file)
                
                # Try again with the fallback
                result = self._memory.add(
                    text, 
                    user_id=memory_id,
                    metadata=metadata
                )
                memory_id = result.get("id")
                logger.info(f"Added memory with fallback system, ID: {memory_id}")
                return memory_id
            
            return None
            
    def get_relevant_memories(self, query: str, memory_id: str = "default", limit: int = 5) -> List[Dict[str, Any]]:
        """
        Get memories relevant to the given query.
        
        Args:
            query: The query text to find relevant memories for
            memory_id: User ID for memory retrieval
            limit: Maximum number of memories to return
            
        Returns:
            List of memory objects with similarity scores
        """
        if not self._initialized:
            self._initialize()
            
        try:
            # Search for relevant memories
            result = self._memory.search(query, user_id=memory_id, limit=limit)
            
            # Format the results
            memories = []
            
            # Handle different result formats based on memory system
            if self.current_mode == self.MEMORY_MODE_MEM0AI:
                # New API version returns a dict with 'results' key
                if isinstance(result, dict) and "results" in result:
                    for memory in result["results"]:
                        memory_obj = {
                            "id": memory.get("id", ""),
                            "text": memory.get("memory", ""),
                            "score": memory.get("score", 0),
                            "metadata": memory.get("metadata", {})
                        }
                        memories.append(memory_obj)
                else:
                    # Legacy fallback
                    for memory in result:
                        memory_obj = {
                            "id": memory.get("id", ""),
                            "text": memory.get("memory", ""),
                            "score": memory.get("score", 0),
                            "metadata": memory.get("metadata", {})
                        }
                        memories.append(memory_obj)
            else:
                # Fallback system returns dict with 'results' key
                if result and "results" in result:
                    for memory in result["results"]:
                        memory_obj = {
                            "id": memory.get("id", ""),
                            "text": memory.get("memory", ""),
                            "score": memory.get("score", 0),
                            "metadata": memory.get("metadata", {})
                        }
                        memories.append(memory_obj)
            
            # Mark memories as active
            for memory in memories:
                self.mark_memory_active(memory.get("id"), memory_id)
                    
            logger.info(f"Retrieved {len(memories)} relevant memories for query")
            return memories
        except Exception as e:
            logger.error(f"Error searching memories: {e}")
            
            # If we got a vector dimension error, fall back to the simpler system
            if "vector dimension error" in str(e).lower() and self.current_mode == self.MEMORY_MODE_MEM0AI:
                logger.warning("Vector dimension mismatch in search. Falling back to simpler store.")
                self.current_mode = self.MEMORY_MODE_FALLBACK
                self._memory = MemoryStore(db_file=self.db_file)
                
                # Try again with the fallback
                result = self._memory.search(query, user_id=memory_id, limit=limit)
                
                # Format results
                memories = []
                if result and "results" in result:
                    for memory in result["results"]:
                        memory_obj = {
                            "id": memory.get("id", ""),
                            "text": memory.get("memory", ""),
                            "score": memory.get("score", 0),
                            "metadata": memory.get("metadata", {})
                        }
                        memories.append(memory_obj)
                        
                # Mark active and return
                for memory in memories:
                    self.mark_memory_active(memory.get("id"), memory_id)
                
                logger.info(f"Retrieved {len(memories)} memories using fallback system")
                return memories
            
            return []
            
    def list_memories(self, memory_id: str = "default", limit: int = 100, offset: int = 0, source: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        List memories for a user with optional filtering by source.
        
        Args:
            memory_id: User ID to get memories for
            limit: Maximum number of memories
            offset: Offset for pagination
            source: Optional filter by source ('mem0ai' or 'fallback')
            
        Returns:
            List of memory objects
        """
        if not self._initialized:
            self._initialize()
            
        try:
            # Get memories from the appropriate system
            all_memories = []
            
            if self.current_mode == self.MEMORY_MODE_MEM0AI:
                # Use get_all method with appropriate parameters
                result = self._memory.get_all(user_id=memory_id)
                
                # Check result format based on memory system version
                if isinstance(result, dict) and "results" in result:
                    all_memories = result["results"]
                else:
                    # Legacy format
                    all_memories = result if isinstance(result, list) else []
            else:
                # Fallback system
                all_memories = self._memory.get_all(user_id=memory_id)
            
            # Apply pagination manually
            paginated_memories = all_memories[offset:offset+limit] if all_memories else []
                
            # Convert to standard format with optional source filtering
            result = []
            for memory in paginated_memories:
                memory_obj = {
                    "id": memory.get("id", ""),
                    "text": memory.get("memory", ""),
                    "metadata": memory.get("metadata", {})
                }
                
                # Check if we need to filter by source
                if source is not None:
                    memory_source = memory_obj.get("metadata", {}).get("source", "unknown")
                    if memory_source != source:
                        continue
                        
                result.append(memory_obj)
                
            logger.info(f"Listed {len(result)} memories for {memory_id}" + 
                      (f" with source filter: {source}" if source else ""))
            return result
        except Exception as e:
            logger.error(f"Error listing memories: {e}")
            
            # If we got a dimension error, fall back to the simpler system
            if "dimension error" in str(e).lower() and self.current_mode == self.MEMORY_MODE_MEM0AI:
                logger.warning("Dimension mismatch in listing. Falling back to simpler store.")
                self.current_mode = self.MEMORY_MODE_FALLBACK
                self._memory = MemoryStore(db_file=self.db_file)
                
                # Try again with the fallback
                all_memories = self._memory.get_all(user_id=memory_id)
                
                # Apply pagination & formatting
                paginated_memories = all_memories[offset:offset+limit] if all_memories else []
                
                result = []
                for memory in paginated_memories:
                    memory_obj = {
                        "id": memory.get("id", ""),
                        "text": memory.get("memory", ""),
                        "metadata": memory.get("metadata", {})
                    }
                    
                    # Apply source filter
                    if source is not None:
                        memory_source = memory_obj.get("metadata", {}).get("source", "unknown")
                        if memory_source != source:
                            continue
                    
                    result.append(memory_obj)
                
                logger.info(f"Listed {len(result)} memories using fallback system")
                return result
            
            return []
    
    def reset_active_memories(self, memory_id: str = "default") -> None:
        """
        Reset active memories for a group.
        
        Args:
            memory_id: User ID to reset active memories for
        """
        if memory_id not in self._active_memories:
            self._active_memories[memory_id] = set()
        else:
            self._active_memories[memory_id].clear()
            
        logger.debug(f"Reset active memories for {memory_id}")
    
    def mark_memory_active(self, memory_item_id: str, memory_id: str = "default") -> None:
        """
        Mark a memory as active for the current conversation.
        
        Args:
            memory_item_id: ID of memory to mark active
            memory_id: User ID of the memory group
        """
        if memory_id not in self._active_memories:
            self._active_memories[memory_id] = set()
            
        self._active_memories[memory_id].add(memory_item_id)
        logger.debug(f"Marked memory {memory_item_id} as active for {memory_id}")
    
    def get_active_memories(self, memory_id: str = "default") -> List[str]:
        """
        Get list of active memory IDs.
        
        Args:
            memory_id: User ID to get active memories for
            
        Returns:
            List of active memory IDs
        """
        if memory_id not in self._active_memories:
            return []
            
        return list(self._active_memories[memory_id])
    
    def get_memory(self, memory_id: str, user_id: str = "default") -> Optional[Dict[str, Any]]:
        """
        Get a specific memory by ID.
        
        Args:
            memory_id: ID of the memory to retrieve
            user_id: User ID of the memory
            
        Returns:
            Memory object if found, None otherwise
        """
        if not self._initialized:
            self._initialize()
            
        try:
            if self.current_mode == self.MEMORY_MODE_MEM0AI:
                # Use get method if available
                if hasattr(self._memory, "get") and callable(getattr(self._memory, "get")):
                    memory = self._memory.get(memory_id)
                    if memory:
                        return {
                            "id": memory.get("id", ""),
                            "text": memory.get("memory", ""),
                            "metadata": memory.get("metadata", {})
                        }
                # Fallback to list+filter approach
                else:
                    memories = self.list_memories(user_id, limit=100)
                    for memory in memories:
                        if memory.get("id") == memory_id:
                            return memory
            else:
                # Fallback list+filter approach
                memories = self.list_memories(user_id, limit=100)
                for memory in memories:
                    if memory.get("id") == memory_id:
                        return memory
                    
            return None
        except Exception as e:
            logger.error(f"Error getting memory {memory_id}: {e}")
            return None
            
    def get_memory_system_info(self) -> Dict[str, Any]:
        """
        Get information about the current memory system state.
        
        Returns:
            Dict with information about the memory system
        """
        return {
            "mode": self.memory_mode,  # What mode was requested
            "current_mode": self.current_mode,  # What mode is actually active
            "mem0ai_available": HAS_MEM0AI,  # Is mem0ai available?
            "using_fallback": self.current_mode == self.MEMORY_MODE_FALLBACK,  # Are we actually using fallback?
            "memory_file": self.db_file if self.current_mode == self.MEMORY_MODE_FALLBACK else None,  # JSON file if using fallback
            "vector_db": self.mem0_host if self.current_mode == self.MEMORY_MODE_MEM0AI else None  # Vector DB if using mem0ai
        }
    
    def set_memory_mode(self, mode: str) -> Dict[str, Any]:
        """
        Change the memory system mode and reinitialize.
        
        Args:
            mode: New memory mode (auto, mem0ai, fallback)
            
        Returns:
            Dict with the result of the operation
        """
        if mode not in [self.MEMORY_MODE_AUTO, self.MEMORY_MODE_MEM0AI, self.MEMORY_MODE_FALLBACK]:
            return {
                "success": False,
                "error": f"Invalid mode: {mode}. Must be one of: {self.MEMORY_MODE_AUTO}, {self.MEMORY_MODE_MEM0AI}, {self.MEMORY_MODE_FALLBACK}"
            }
            
        try:
            # Check if mem0ai is requested but not available
            if mode == self.MEMORY_MODE_MEM0AI and not HAS_MEM0AI:
                logger.warning(f"mem0ai mode requested but mem0ai is not available")
                return {
                    "success": False,
                    "error": "mem0ai is not available. Please install mem0ai or use a different mode.",
                    "mode": self.memory_mode,
                    "current_mode": self.current_mode,
                    "mem0ai_available": HAS_MEM0AI
                }
            
            # Save old values to restore in case of failure
            old_mode = self.memory_mode
            old_memory = self._memory
            old_initialized = self._initialized
            old_current_mode = self.current_mode
            
            # Set new mode and reinitialize
            self.memory_mode = mode
            self._initialized = False
            self._initialize()
            
            return {
                "success": True,
                "mode": self.memory_mode,
                "current_mode": self.current_mode,
                "using_fallback": self.current_mode == self.MEMORY_MODE_FALLBACK,
                "mem0ai_available": HAS_MEM0AI
            }
        except Exception as e:
            # Restore old values if something went wrong
            self.memory_mode = old_mode
            self._memory = old_memory
            self._initialized = old_initialized
            self.current_mode = old_current_mode
            
            return {
                "success": False,
                "error": str(e),
                "mode": self.memory_mode,
                "current_mode": self.current_mode,
                "mem0ai_available": HAS_MEM0AI
            }
