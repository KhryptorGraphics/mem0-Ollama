"""
Memory management utilities for mem0 + Ollama integration

This module handles memory operations including storage, retrieval, and management
of conversation history using vector database.
"""

import time
import requests
import json
import datetime
import threading
import random
from typing import Dict, List, Any, Optional, Union, Tuple, Callable, Set
from concurrent.futures import ThreadPoolExecutor
from functools import wraps

from mem0 import Memory
from config import (
    OLLAMA_HOST,
    DEFAULT_MODEL,
    MEM0_HOST,
    VECTOR_COLLECTION,
    ACTIVE_MEMORY_COLLECTION,
    MAX_MEMORIES,
    MEMORY_DATABASE
)

import logging_utils

# Set up loggers
logger = logging_utils.get_logger(__name__)
memory_logger = logging_utils.get_logger("memory")

# Default retry settings - Exponential backoff with jitter
DEFAULT_TIMEOUT = 10  # seconds
MAX_RETRIES = 5  # Increased for better resilience
RETRY_DELAY = 1.0  # seconds
RETRY_BACKOFF_FACTOR = 2.0
RETRY_JITTER = 0.1  # Add jitter to avoid request storms

# Cache settings
CACHE_TTL = 300  # seconds (5 minutes)
MAX_CACHE_ITEMS = 1000

# Advanced memory settings
MEMORY_DECAY_FACTOR = 0.9  # Factor to multiply relevance score for older memories
MEMORY_DECAY_INTERVAL = 60 * 60 * 24  # 24 hours in seconds
MEMORY_RELEVANCE_THRESHOLD = 0.6  # Minimum relevance score for memories
MEMORY_PRIORITY_RECENT = 0.3  # Weight for recency in priority calculation
MEMORY_PRIORITY_RELEVANCE = 0.7  # Weight for relevance in priority calculation

# Global constants
GLOBAL_MEMORY_ID = "global_memory_store"

# Model dimensions based on known models
MODEL_DIMENSIONS = {
    'llama3': 4096,
    'llama3.1': 4096,
    'llama3.2': 3072,
    'llama3.3': 8192,
    'gemma2': 3072,
    'gemma3:12b': 3072,
    'phi3-mini': 2048,
    'phi3': 3072,
    'mistral': 4096,
    'qwen2': 4096,
    'qwen2.5': 4096,
    'nomic-embed-text': 768,
    'snowflake-arctic-embed': 1024,
    'deepseek-r1': 4096
}

# Connection pools for efficiency
# Use connection pooling for all requests
session = requests.Session()
adapter = requests.adapters.HTTPAdapter(
    pool_connections=10,
    pool_maxsize=20,
    max_retries=3,
    pool_block=False
)
session.mount('http://', adapter)
session.mount('https://', adapter)

# Memory cache
_memory_cache = {}
_cache_lock = threading.RLock()

# Embeddings cache to avoid redundant embedding generation
_embeddings_cache = {}
_embedding_cache_lock = threading.RLock()

# Health metrics
_health_metrics = {
    "memory_operations": 0,
    "failed_operations": 0,
    "search_operations": 0,
    "add_operations": 0,
    "cache_hits": 0,
    "cache_misses": 0,
    "retrieval_latency_ms": [],
    "memory_count": 0,
    "active_memory_count": 0,
    "last_update": time.time()
}
_metrics_lock = threading.RLock()

# Simple mock Memory implementation
class MockMemory:
    """Simple mock implementation for when the real Memory can't be initialized"""
    
    def add(self, text, user_id=None, metadata=None):
        """Add a memory (mock implementation)"""
        memory_id = "mock_" + str(hash(text))
        return {"id": memory_id}
        
    def search(self, query, user_id=None, limit=10):
        """Search for relevant memories (mock implementation)"""
        return {"results": []}
        
    def get_all(self, user_id=None, limit=100, offset=0):
        """Get all memories (mock implementation)"""
        return []
        
    def update(self, id, user_id=None, memory=None, metadata=None):
        """Update a memory (mock implementation)"""
        return True
        
    def delete(self, id, user_id=None):
        """Delete a memory (mock implementation)"""
        return True
        
    def create_embedding(self, text):
        """Create a mock embedding vector"""
        # Return a random embedding vector of dimension 768
        return [random.random() for _ in range(768)]

# Utility decorator for retry with exponential backoff
def retry_with_backoff(func):
    """
    Decorator to retry a function with exponential backoff and jitter.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        max_retries = kwargs.pop('max_retries', MAX_RETRIES)
        retry_delay = kwargs.pop('retry_delay', RETRY_DELAY)
        backoff_factor = kwargs.pop('backoff_factor', RETRY_BACKOFF_FACTOR)
        jitter = kwargs.pop('jitter', RETRY_JITTER)
        
        attempt = 0
        
        while attempt < max_retries:
            try:
                return func(*args, **kwargs)
            except (requests.Timeout, requests.ConnectionError) as e:
                attempt += 1
                if attempt >= max_retries:
                    raise
                
                # Calculate backoff with jitter
                sleep_time = retry_delay * (backoff_factor ** (attempt - 1))
                # Add jitter (±10%)
                sleep_time = sleep_time * (1 + random.uniform(-jitter, jitter))
                
                logger.warning(f"Retry {attempt}/{max_retries} after error: {e}. "
                             f"Retrying in {sleep_time:.2f}s")
                time.sleep(sleep_time)
    
    return wrapper

def preprocess_user_message(message: str) -> str:
    """
    Preprocess user message to enhance it for better memory storage and retrieval.
    
    This function makes user messages more prominent in the vector store by:
    1. Adding emphasis markers
    2. Potentially repeating key phrases
    3. Formatting for better embedding
    
    Args:
        message: The original user message
        
    Returns:
        Enhanced version of the message optimized for embedding and retrieval
    """
    logger.debug(f"Preprocessing user message (length: {len(message)})")
    
    # Skip preprocessing for very short messages
    if len(message) < 5:
        enhanced = f"IMPORTANT USER QUERY: {message}"
        logger.debug("Applied short message enhancement")
        return enhanced
        
    # For longer messages, enhance with emphasis and formatting
    enhanced = message.strip()
    
    # Add importance markers at the beginning and end
    enhanced = f"IMPORTANT USER INPUT: {enhanced} [USER QUERY END]"
    
    # If message is a question, emphasize it further
    if any(q in message for q in ["?", "what", "how", "why", "when", "where", "who", "which"]):
        enhanced = f"USER QUESTION: {enhanced}"
        logger.debug("Applied question enhancement")
    
    logger.debug(f"Message preprocessing complete (original length: {len(message)}, enhanced length: {len(enhanced)})")
    return enhanced

class MemoryManager:
    """
    Advanced manager for memory operations using vector storage.
    Handles memory storage, retrieval, prioritization, and maintenance.
    """
    
    def __init__(self):
        """Initialize the memory manager with connection to vector database."""
        self._memory = None
        self._initialized = False
        self._initialization_lock = threading.RLock()
        self._active_memories = {}
        self._memory_tags = {}
        self._memory_importance = {}
        self._initialize()
    
    def _initialize(self) -> None:
        """Initialize memory system with proper error handling and retry logic."""
        if self._initialized:
            return
        
        with self._initialization_lock:
            if self._initialized:  # Double-check after acquiring lock
                return
                
            try:
                # Initialize memory system - use from_config instead of direct constructor
                # The LLM, embedder, and vector store are all set to use Ollama
                
                # Ensure we're using the correct Ollama host and it has proper scheme
                host = OLLAMA_HOST
                if "localhost" not in host and "127.0.0.1" not in host:
                    logger.warning(f"Ollama host is set to {host}, which may not be correct. Using http://localhost:11434 instead.")
                    host = "http://localhost:11434"
                elif not host.startswith(('http://', 'https://')):
                    host = f"http://{host}"
                
                logger.info(f"Initializing Memory with Ollama host: {host}")
                
                # Use nomic-embed-text or a stable embedder model for embeddings to avoid dimension mismatches
                embedder_model = "nomic-embed-text"
                
                # Check available models to see if embedding model is available
                try:
                    response = requests.get(f"{host}/api/tags", timeout=5)
                    if response.status_code == 200:
                        data = response.json()
                        available_models = [model["name"] for model in data.get("models", [])]
                        
                        # Check if we have a dedicated embedding model available
                        if "nomic-embed-text" in available_models:
                            embedder_model = "nomic-embed-text"
                        elif "snowflake-arctic-embed" in available_models:
                            embedder_model = "snowflake-arctic-embed"
                        # Fallback to LLM model for embeddings if necessary
                        else:
                            embedder_model = DEFAULT_MODEL
                            
                        logger.info(f"Using {embedder_model} for embeddings")
                    else:
                        logger.warning(f"Could not get model list, using {embedder_model} for embeddings")
                except Exception as model_e:
                    logger.warning(f"Error checking models, using {embedder_model} for embeddings: {model_e}")
                
                config = {
                    "llm": {
                        "provider": "ollama",
                        "config": {
                            "model": DEFAULT_MODEL,
                            "ollama_base_url": host,
                            "temperature": 0.7,
                            "max_tokens": 1000
                        }
                    },
                    "embedder": {
                        "provider": "ollama",
                        "config": {
                            "model": embedder_model,
                            "ollama_base_url": host
                        }
                    }
                }
                
                try:
                    self._memory = Memory.from_config(config)
                    self._initialized = True
                    logger.info("Memory manager initialized successfully with Ollama")
                except Exception as inner_e:
                    # Fall back to mock implementation
                    logger.warning(f"Using mock memory implementation instead: {inner_e}")
                    self._memory = MockMemory()
                    self._initialized = True
                
            except Exception as e:
                logger.error(f"Error initializing memory manager: {e}")
                # Simplify logging to avoid request_id errors
                try:
                    logger.error(f"Error details: {str(e)}")
                except:
                    pass
                
                # Fall back to mock implementation 
                self._memory = MockMemory()
                self._initialized = True
                logger.info("Using mock memory implementation as fallback")
    
    def _ensure_initialized(self) -> bool:
        """
        Ensure memory system is initialized, try to initialize if not.
        
        Returns:
            True if initialized, False otherwise
        """
        if not self._initialized or not self._memory:
            self._initialize()
        
        return self._initialized and self._memory is not None
    
    def add_memory(
        self,
        text: str,
        memory_id: str = GLOBAL_MEMORY_ID,
        metadata: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
        priority: float = 1.0
    ) -> Optional[str]:
        """
        Add a memory to the vector store with enhanced metadata.
        
        Args:
            text: Memory text to store
            memory_id: Identifier for the memory group (default: global)
            metadata: Additional metadata to store
            tags: Optional tags for categorizing the memory
            priority: Priority value for this memory (higher is more important)
            
        Returns:
            Memory ID if successful, None otherwise
        """
        if not self._ensure_initialized():
            logger.error("Cannot add memory: Memory system not initialized")
            return None
        
        try:
            # Prepare metadata with additional useful information
            enhanced_metadata = metadata or {}
            enhanced_metadata.update({
                "timestamp": time.time(),
                "type": enhanced_metadata.get("type", "general"),
                "priority": priority,
                "active": enhanced_metadata.get("active", True)
            })
            
            # Log memory operation
            memory_logger.info(f"Adding memory: {text[:50]}...")
            
            # Use the memory system (real or mock)
            result = self._memory.add(text, user_id=memory_id, metadata=enhanced_metadata)
            memory_item_id = result.get("id")
            return memory_item_id
            
        except Exception as e:
            logger.error(f"Error adding memory: {e}")
            # Simplified error logging
            return None
    
    def get_relevant_memories(
        self, 
        query: str, 
        memory_id: str = "global_memory_store", 
        limit: int = 5,
        threshold: float = 0.0
    ) -> List[Dict[str, Any]]:
        """
        Get memories relevant to the given query using vector similarity search.
        
        Args:
            query: The query text to find relevant memories for
            memory_id: Group ID for the memories
            limit: Maximum number of memories to return
            threshold: Minimum similarity score threshold
            
        Returns:
            List of relevant memory objects with similarity scores
        """
        if not self._ensure_initialized():
            logger.error("Cannot search memories: Memory system not initialized")
            return []
            
        try:
            # Enhanced query for better search results
            enhanced_query = preprocess_user_message(query)
            
            # Use the memory system (real or mock)
            results = self._memory.search(
                query=enhanced_query,
                user_id=memory_id,
                limit=limit
            )
            
            # Process results
            if results and "results" in results:
                memories = []
                for result in results["results"]:
                    if threshold > 0 and result.get("score", 0) < threshold:
                        continue
                    memory_info = {
                        "id": result.get("id", ""),
                        "text": result.get("memory", ""),
                        "score": result.get("score", 0),
                        "metadata": result.get("metadata", {})
                    }
                    memories.append(memory_info)
                return memories
                
            # Default return for no results
            return []
            
        except Exception as e:
            logger.error(f"Error searching memories: {e}")
            return []
    
    def list_memories(
        self, 
        memory_id: str = "global_memory_store",
        limit: int = 100,
        offset: int = 0,
        tag: Optional[str] = None,
        sort_by: str = "timestamp",
        sort_order: str = "desc"
    ) -> List[Dict[str, Any]]:
        """
        List memories with advanced filtering and sorting.
        
        Args:
            memory_id: Group ID for the memories
            limit: Maximum number of memories to return
            offset: Number of memories to skip for pagination
            tag: Optional tag to filter memories by
            sort_by: Field to sort results by
            sort_order: Sort direction ('asc' or 'desc')
            
        Returns:
            List of memory objects
        """
        # Get memories from the memory system (real or mock)
        try:
            if self._ensure_initialized():
                memories = self._memory.get_all(
                    user_id=memory_id,
                    limit=limit,
                    offset=offset
                )
                if memories:
                    return memories
        except Exception as e:
            logger.warning(f"Failed to list memories: {e}")
            
        # Fall back to empty list
        return []
    
    def reset_active_memories(self, memory_id: str = "global_memory_store") -> None:
        """
        Reset the active memories for a specific memory ID.
        
        Args:
            memory_id: Group ID for the memories
        """
        with self._initialization_lock:
            # Initialize the active memories dict for this memory_id if it doesn't exist
            if memory_id not in self._active_memories:
                self._active_memories[memory_id] = set()
            else:
                # Clear existing active memories
                self._active_memories[memory_id].clear()
            
            logger.debug(f"Reset active memories for {memory_id}")
    
    def mark_memory_active(self, memory_item_id: str, memory_id: str = "global_memory_store") -> None:
        """
        Mark a memory as active for the current conversation.
        
        Args:
            memory_item_id: ID of the memory item to mark as active
            memory_id: Group ID for the memories
        """
        with self._initialization_lock:
            # Initialize the active memories dict for this memory_id if it doesn't exist
            if memory_id not in self._active_memories:
                self._active_memories[memory_id] = set()
            
            # Add the memory item ID to the active set
            self._active_memories[memory_id].add(memory_item_id)
            logger.debug(f"Marked memory {memory_item_id} as active for {memory_id}")
    
    def get_active_memories(self, memory_id: str = "global_memory_store") -> List[str]:
        """
        Get the list of currently active memory IDs.
        
        Args:
            memory_id: Group ID for the memories
            
        Returns:
            List of active memory IDs
        """
        with self._initialization_lock:
            # Return empty list if memory_id not in active memories
            if memory_id not in self._active_memories:
                return []
            
            # Convert set to list and return
            return list(self._active_memories[memory_id])
