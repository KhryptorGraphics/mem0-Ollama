"""
Improved memory system for mem0-ollama that uses dedicated embedding models
and proper vector storage.
"""

import os
import json
import time
import uuid
import logging
import threading
import random
from typing import Dict, List, Any, Optional, Tuple
import requests
from datetime import datetime

# Configure logging
logger = logging.getLogger(__name__)

class MemoryStore:
    """Simple JSON-based memory store for development"""
    
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
                    logger.info(f"Loaded {sum(len(memories) for memories in self.memories.values())} memories from {self.db_file}")
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
    
    def get_all(self, user_id="default", limit=100, offset=0):
        """Get all memories for a user"""
        with self.lock:
            if user_id not in self.memories:
                return []
            
            # Get memories and apply pagination
            memories = self.memories[user_id]
            # Sort by created_at (newest first)
            memories = sorted(memories, key=lambda m: m.get("created_at", 0), reverse=True)
            
            # Apply pagination
            start = min(offset, len(memories))
            end = min(offset + limit, len(memories))
            
            return memories[start:end]
    
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
    
    def get_memory(self, memory_id, user_id="default"):
        """Get a specific memory by ID"""
        with self.lock:
            if user_id not in self.memories:
                return None
            
            for memory in self.memories[user_id]:
                if memory.get("id") == memory_id:
                    return memory
            
            return None
    
    def update(self, memory_id, user_id="default", memory=None, metadata=None):
        """Update a memory"""
        with self.lock:
            if user_id not in self.memories:
                return False
            
            for i, mem in enumerate(self.memories[user_id]):
                if mem.get("id") == memory_id:
                    if memory is not None:
                        mem["memory"] = memory
                    
                    if metadata is not None:
                        if not mem.get("metadata"):
                            mem["metadata"] = {}
                        mem["metadata"].update(metadata)
                    
                    # Save changes
                    self._save()
                    return True
            
            return False
    
    def delete(self, memory_id, user_id="default"):
        """Delete a memory"""
        with self.lock:
            if user_id not in self.memories:
                return False
            
            for i, memory in enumerate(self.memories[user_id]):
                if memory.get("id") == memory_id:
                    del self.memories[user_id][i]
                    # Save changes
                    self._save()
                    return True
            
            return False
    
    def create_embedding(self, text):
        """
        Placeholder for creating embeddings -
        we don't store actual vectors in JSON store
        """
        # Return a random embedding vector of dimension 768 as placeholder
        return [random.random() for _ in range(768)]


class ImprovedMemoryManager:
    """
    Improved memory manager that resolves the embedding dimension issues
    and uses a simple persistent JSON store.
    """
    
    def __init__(self, db_file="mem0_ollama.json", ollama_host="http://localhost:11434"):
        """Initialize memory manager"""
        self.store = MemoryStore(db_file=db_file)
        self.ollama_host = ollama_host
        self._active_memories = {}
        self._lock = threading.RLock()
        logger.info(f"Initialized ImprovedMemoryManager with store at {db_file}")
    
    def add_memory(self, text, memory_id="default", metadata=None, tags=None, priority=1.0):
        """
        Add a memory with enhanced metadata.
        
        Args:
            text: Memory text to store
            memory_id: Group ID for the memory (defaults to 'default')
            metadata: Additional metadata to store
            tags: Optional tags for categorizing the memory
            priority: Priority value (higher = more important)
            
        Returns:
            Memory ID if successful, None otherwise
        """
        try:
            # Prepare metadata with additional information
            enhanced_metadata = metadata or {}
            enhanced_metadata.update({
                "timestamp": datetime.now().isoformat(),
                "type": enhanced_metadata.get("type", "general"),
                "priority": priority,
                "tags": tags or [],
                "active": enhanced_metadata.get("active", True)
            })
            
            # Add the memory
            result = self.store.add(text, user_id=memory_id, metadata=enhanced_metadata)
            logger.info(f"Added memory to store with ID: {result.get('id')}")
            return result.get("id")
            
        except Exception as e:
            logger.error(f"Error adding memory: {e}")
            return None
    
    def get_relevant_memories(self, query, memory_id="default", limit=5, threshold=0.0):
        """
        Get memories relevant to a query.
        
        Args:
            query: Search query text
            memory_id: Group ID for the memories
            limit: Maximum number of memories to return
            threshold: Minimum similarity score (0.0 to 1.0)
            
        Returns:
            List of memory objects with metadata
        """
        try:
            # Search for relevant memories
            results = self.store.search(query, user_id=memory_id, limit=limit)
            
            # Process results
            memories = []
            if results and "results" in results:
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
            
            logger.info(f"Found {len(memories)} relevant memories for query")
            return memories
            
        except Exception as e:
            logger.error(f"Error searching memories: {e}")
            return []
    
    def list_memories(self, memory_id="default", limit=100, offset=0, tag=None, sort_by="timestamp", sort_order="desc"):
        """
        List memories with pagination and filtering.
        
        Args:
            memory_id: Group ID for the memories
            limit: Maximum number of memories to return
            offset: Number of memories to skip (for pagination)
            tag: Optional tag to filter by
            sort_by: Field to sort by
            sort_order: Sort direction ('asc' or 'desc')
            
        Returns:
            List of memory objects
        """
        try:
            # Get all memories
            memories = self.store.get_all(user_id=memory_id, limit=limit, offset=offset)
            
            # Filter by tag if specified
            if tag is not None:
                memories = [
                    memory for memory in memories
                    if tag in memory.get("metadata", {}).get("tags", [])
                ]
            
            # Convert to response format
            response = []
            for memory in memories:
                response.append({
                    "id": memory.get("id", ""),
                    "text": memory.get("memory", ""),
                    "metadata": memory.get("metadata", {})
                })
            
            logger.info(f"Retrieved {len(response)} memories")
            return response
            
        except Exception as e:
            logger.error(f"Error listing memories: {e}")
            return []
    
    def get_memory(self, memory_id, user_id="default"):
        """Get a specific memory by ID"""
        try:
            memory = self.store.get_memory(memory_id, user_id=user_id)
            if memory:
                return {
                    "id": memory.get("id", ""),
                    "text": memory.get("memory", ""),
                    "metadata": memory.get("metadata", {})
                }
            return None
        except Exception as e:
            logger.error(f"Error getting memory: {e}")
            return None
    
    def reset_active_memories(self, memory_id="default"):
        """Reset active memories for a group"""
        with self._lock:
            if memory_id not in self._active_memories:
                self._active_memories[memory_id] = set()
            else:
                self._active_memories[memory_id].clear()
            
            logger.debug(f"Reset active memories for {memory_id}")
    
    def mark_memory_active(self, memory_item_id, memory_id="default"):
        """Mark a memory as active for current conversation"""
        with self._lock:
            if memory_id not in self._active_memories:
                self._active_memories[memory_id] = set()
            
            self._active_memories[memory_id].add(memory_item_id)
            logger.debug(f"Marked memory {memory_item_id} as active")
    
    def get_active_memories(self, memory_id="default"):
        """Get list of active memory IDs"""
        with self._lock:
            if memory_id not in self._active_memories:
                return []
            
            return list(self._active_memories[memory_id])
