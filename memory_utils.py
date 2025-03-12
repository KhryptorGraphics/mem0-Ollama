import os
import json
import requests
import logging
import uuid
from config import *
import time
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

class MemoryManager:
    def __init__(self):
        self.mem0_host = MEM0_HOST
        self.vector_collection = VECTOR_COLLECTION
        self.active_memory_collection = ACTIVE_MEMORY_COLLECTION
        self.initialize()
        logger.info(f"MemoryManager initialized with host: {self.mem0_host}")

    def initialize(self):
        """Initialize the memory database if it doesn't exist"""
        try:
            # Ensure collections exist
            self._ensure_collection_exists(self.vector_collection)
            self._ensure_collection_exists(self.active_memory_collection)
            logger.info("Memory collections initialized")
        except Exception as e:
            logger.error(f"Error initializing memory database: {e}")

    def _ensure_collection_exists(self, collection_name):
        """Create a collection if it doesn't exist"""
        try:
            # Check if collection exists
            response = requests.get(f"{self.mem0_host}/api/collections/{collection_name}")
            
            if response.status_code == 404:
                # Create the collection
                create_response = requests.post(
                    f"{self.mem0_host}/api/collections",
                    json={"name": collection_name, "metadata": {"description": f"Collection for {collection_name}"}}
                )
                create_response.raise_for_status()
                logger.info(f"Created collection: {collection_name}")
            elif response.status_code != 200:
                response.raise_for_status()
                
        except Exception as e:
            logger.error(f"Error ensuring collection {collection_name} exists: {e}")
            raise

    def add_memory(self, text, memory_id='default', metadata=None):
        """Add a memory to the database"""
        try:
            if metadata is None:
                metadata = {}
            
            # Add timestamp if not present
            if 'timestamp' not in metadata:
                metadata['timestamp'] = datetime.now().isoformat()
            
            # Add memory_id to metadata
            metadata['memory_id'] = memory_id
            
            # Generate a unique ID
            memory_item_id = str(uuid.uuid4())
            metadata['id'] = memory_item_id
            
            # Create the memory in mem0
            response = requests.post(
                f"{self.mem0_host}/api/collections/{self.vector_collection}/items",
                json={
                    "id": memory_item_id,
                    "text": text,
                    "metadata": metadata
                }
            )
            response.raise_for_status()
            logger.info(f"Added memory with ID: {memory_item_id}")
            return memory_item_id
        except Exception as e:
            logger.error(f"Error adding memory: {e}")
            return None

    def get_memory(self, memory_item_id, memory_id='default'):
        """Get a specific memory by ID"""
        try:
            response = requests.get(
                f"{self.mem0_host}/api/collections/{self.vector_collection}/items/{memory_item_id}"
            )
            response.raise_for_status()
            memory = response.json()
            
            # Verify memory belongs to the right memory_id
            if memory.get('metadata', {}).get('memory_id') != memory_id:
                return None
                
            return memory
        except Exception as e:
            logger.error(f"Error getting memory {memory_item_id}: {e}")
            return None

    def get_relevant_memories(self, query, memory_id='default', limit=5):
        """Retrieve memories relevant to a query"""
        try:
            response = requests.post(
                f"{self.mem0_host}/api/collections/{self.vector_collection}/search",
                json={
                    "query": query,
                    "filter": {"memory_id": memory_id},
                    "limit": limit
                }
            )
            response.raise_for_status()
            results = response.json()
            logger.info(f"Retrieved {len(results.get('items', []))} relevant memories for query")
            return results.get('items', [])
        except Exception as e:
            logger.error(f"Error getting relevant memories: {e}")
            return []

    def list_memories(self, memory_id='default', limit=100, offset=0):
        """List all memories for a given memory ID"""
        try:
            # Get all memories with the specific memory_id
            response = requests.post(
                f"{self.mem0_host}/api/collections/{self.vector_collection}/search",
                json={
                    "query": "",  # Empty query to get all
                    "filter": {"memory_id": memory_id},
                    "limit": limit,
                    "offset": offset
                }
            )
            response.raise_for_status()
            results = response.json()
            return results.get('items', [])
        except Exception as e:
            logger.error(f"Error listing memories: {e}")
            return []

    def mark_memory_active(self, memory_item_id, memory_id='default'):
        """Mark a memory as active for the current context"""
        try:
            # Get the memory first to verify it exists
            memory = self.get_memory(memory_item_id, memory_id)
            if not memory:
                logger.warning(f"Cannot mark non-existent memory {memory_item_id} as active")
                return False
                
            # Create active memory record
            active_id = f"{memory_id}_{memory_item_id}"
            
            # Check if already exists
            try:
                response = requests.get(
                    f"{self.mem0_host}/api/collections/{self.active_memory_collection}/items/{active_id}"
                )
                if response.status_code == 200:
                    logger.info(f"Memory {memory_item_id} is already active")
                    return True
            except:
                pass  # If not found, we'll create it
                
            # Add to active memories
            response = requests.post(
                f"{self.mem0_host}/api/collections/{self.active_memory_collection}/items",
                json={
                    "id": active_id,
                    "text": memory.get('text', ''),
                    "metadata": {
                        "memory_id": memory_id,
                        "memory_item_id": memory_item_id,
                        "timestamp": datetime.now().isoformat()
                    }
                }
            )
            response.raise_for_status()
            logger.info(f"Marked memory {memory_item_id} as active")
            return True
        except Exception as e:
            logger.error(f"Error marking memory as active: {e}")
            return False

    def reset_active_memories(self, memory_id='default'):
        """Reset active memories for a given memory ID"""
        try:
            # Search for active memories with the specific memory_id
            response = requests.post(
                f"{self.mem0_host}/api/collections/{self.active_memory_collection}/search",
                json={
                    "query": "",  # Empty query to get all
                    "filter": {"memory_id": memory_id},
                    "limit": 1000  # Large limit to get all
                }
            )
            response.raise_for_status()
            results = response.json()
            
            # Delete each active memory
            for item in results.get('items', []):
                item_id = item.get('id')
                if item_id:
                    delete_response = requests.delete(
                        f"{self.mem0_host}/api/collections/{self.active_memory_collection}/items/{item_id}"
                    )
                    delete_response.raise_for_status()
            
            logger.info(f"Reset {len(results.get('items', []))} active memories for {memory_id}")
            return True
        except Exception as e:
            logger.error(f"Error resetting active memories: {e}")
            return False

    def get_active_memories(self, memory_id='default'):
        """Get all active memories for a given memory ID"""
        try:
            response = requests.post(
                f"{self.mem0_host}/api/collections/{self.active_memory_collection}/search",
                json={
                    "query": "",  # Empty query to get all
                    "filter": {"memory_id": memory_id},
                    "limit": 1000  # Large limit to get all
                }
            )
            response.raise_for_status()
            results = response.json()
            
            # Extract memory_item_ids from the results
            active_memory_ids = [item.get('metadata', {}).get('memory_item_id') 
                              for item in results.get('items', [])]
            logger.info(f"Retrieved {len(active_memory_ids)} active memories")
            return active_memory_ids
        except Exception as e:
            logger.error(f"Error getting active memories: {e}")
            return []
