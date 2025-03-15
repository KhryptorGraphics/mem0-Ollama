"""
Mock implementation of the Qdrant MCP module for testing purposes

This module simulates the Qdrant MCP tool functionality, providing mock implementations
of the store and find operations that will be called via the MCP tool.
"""

import time
import json
import os
import logging
from typing import Dict, List, Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# In-memory storage for mock vector database
MOCK_MEMORY_STORE = []
MOCK_STORAGE_FILE = "qdrant_mcp_mock_storage.json"

# Load existing data if available
def _load_storage():
    """Load mock storage from file if it exists"""
    global MOCK_MEMORY_STORE
    
    try:
        if os.path.exists(MOCK_STORAGE_FILE):
            with open(MOCK_STORAGE_FILE, 'r') as f:
                MOCK_MEMORY_STORE = json.load(f)
                logger.info(f"Loaded {len(MOCK_MEMORY_STORE)} items from mock storage file")
    except Exception as e:
        logger.error(f"Error loading mock storage: {e}")
        MOCK_MEMORY_STORE = []

# Save data to file
def _save_storage():
    """Save mock storage to file"""
    try:
        with open(MOCK_STORAGE_FILE, 'w') as f:
            json.dump(MOCK_MEMORY_STORE, f, indent=2)
    except Exception as e:
        logger.error(f"Error saving mock storage: {e}")

# Load the storage on module import
_load_storage()

def store(information: str, metadata: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Mock implementation of the qdrant-store MCP tool
    
    Args:
        information: Text to store
        metadata: Additional metadata to store
        
    Returns:
        Dict with storage result
    """
    try:
        # Create a unique ID
        item_id = f"item_{int(time.time())}_{len(MOCK_MEMORY_STORE)}"
        
        # Create the item to store
        item = {
            "id": item_id,
            "text": information,
            "metadata": metadata or {},
            "timestamp": time.time()
        }
        
        # Store in our mock database
        MOCK_MEMORY_STORE.append(item)
        
        # Save to file
        _save_storage()
        
        logger.info(f"Stored item with ID: {item_id}")
        
        # Return success result
        return {
            "id": item_id,
            "success": True
        }
    except Exception as e:
        logger.error(f"Error in store operation: {e}")
        return {
            "success": False,
            "error": str(e)
        }

def find(query: str) -> Dict[str, Any]:
    """
    Mock implementation of the qdrant-find MCP tool
    
    Args:
        query: Text to search for
        
    Returns:
        Dict with search results
    """
    try:
        # Simple keyword search in our mock database
        results = []
        query_lower = query.lower()
        
        for item in MOCK_MEMORY_STORE:
            text = item.get("text", "").lower()
            
            # Calculate a score based on keyword matches
            if query_lower in text:
                # Create a score based on how close the match is
                score = 0.8
            else:
                # Check for partial word matches
                query_words = query_lower.split()
                text_words = text.split()
                
                # Count matching words
                matching_words = set(query_words).intersection(set(text_words))
                if matching_words:
                    score = 0.5 * len(matching_words) / len(query_words)
                else:
                    # No match
                    score = 0
            
            # Only include results with scores above threshold
            if score > 0.1:
                results.append({
                    "id": item.get("id"),
                    "text": item.get("text"),
                    "metadata": item.get("metadata", {}),
                    "score": score
                })
        
        # Sort by score
        results.sort(key=lambda x: x.get("score", 0), reverse=True)
        
        logger.info(f"Found {len(results)} results for query: {query}")
        
        return {
            "results": results,
            "success": True
        }
    except Exception as e:
        logger.error(f"Error in find operation: {e}")
        return {
            "results": [],
            "success": False,
            "error": str(e)
        }
