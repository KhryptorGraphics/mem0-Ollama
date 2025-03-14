"""
Test script for the integrated mem0ai with Qdrant vector database

This script tests the integration of:
1. Final mem0 adapter
2. Qdrant vector database
3. API endpoints for vector operations
"""

import os
import sys
import json
import time
import requests
import logging
from typing import Dict, Any, List

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import the health check module to use its utility functions
from health_check import check_ollama_health, check_qdrant_health, check_mem0_health
from final_mem0_adapter import Mem0Adapter
from config import OLLAMA_HOST, MEM0_HOST, DEFAULT_MODEL

# API endpoint (assuming the Flask app is running)
API_BASE_URL = "http://localhost:8000/api"

def test_services_availability():
    """Test that all required services are up and running"""
    logger.info("Testing service availability...")
    
    # Check Ollama
    ollama_status = check_ollama_health()
    logger.info(f"Ollama: {ollama_status}")
    if not ollama_status.available:
        logger.error("Ollama is not available. Tests will likely fail.")
        return False
    
    # Check mem0ai
    mem0_status = check_mem0_health()
    logger.info(f"mem0ai: {mem0_status}")
    if not mem0_status.available:
        logger.warning("mem0ai is not available. Will use fallback implementation.")
        
    # Check Qdrant
    qdrant_status = check_qdrant_health()
    logger.info(f"Qdrant: {qdrant_status}")
    if not qdrant_status.available:
        logger.error("Qdrant is not available. Vector operations will fail.")
        return False
    
    return True

def test_api_endpoints():
    """Test the vector-related API endpoints"""
    logger.info("Testing API endpoints...")
    
    try:
        # Test the vector status endpoint
        response = requests.get(f"{API_BASE_URL}/vector/status")
        if response.status_code == 200:
            vector_status = response.json()
            logger.info(f"Vector status: {json.dumps(vector_status, indent=2)}")
            
            # Check if using mem0ai with Qdrant
            if vector_status.get("using_mem0ai", False):
                logger.info("Successfully using mem0ai with vector database")
            else:
                logger.warning("Not using mem0ai with vector database. Using fallback implementation.")
                
            # Get embedding dimensions
            logger.info(f"Embedding dimensions: {vector_status.get('embedding_dimensions', 'unknown')}")
        else:
            logger.error(f"Failed to get vector status: {response.status_code} {response.text}")
            return False
            
        # Test the health check endpoints
        response = requests.get(f"{API_BASE_URL}/health/qdrant")
        if response.status_code == 200:
            qdrant_health = response.json()
            logger.info(f"Qdrant health check via API: {'Available' if qdrant_health.get('available', False) else 'Unavailable'}")
        else:
            logger.error(f"Failed to get Qdrant health: {response.status_code} {response.text}")
            return False
            
        return True
    except Exception as e:
        logger.error(f"Error testing API endpoints: {e}")
        return False

def test_memory_operations():
    """Test memory operations with the adapter directly"""
    logger.info("Testing memory operations using mem0 adapter...")
    
    # Initialize the adapter
    adapter = Mem0Adapter(
        ollama_host=OLLAMA_HOST,
        mem0_host=MEM0_HOST,
        default_model=DEFAULT_MODEL
    )
    
    # Check if we're using mem0ai or fallback
    mem_info = adapter.get_memory_system_info()
    logger.info(f"Memory system info: {mem_info}")
    
    # Set a test user ID
    test_user_id = f"test_vector_{int(time.time())}"
    
    # Step 1: Add a memory
    test_memory = f"This is a test memory created at {time.strftime('%Y-%m-%d %H:%M:%S')}"
    logger.info(f"Adding test memory for user {test_user_id}")
    memory_id = adapter.add_memory(test_memory, test_user_id)
    
    if not memory_id:
        logger.error("Failed to add memory")
        return False
    
    logger.info(f"Successfully added memory with ID: {memory_id}")
    
    # Step 2: Retrieve memories
    logger.info("Listing all memories")
    memories = adapter.list_memories(test_user_id)
    logger.info(f"Found {len(memories)} memories for {test_user_id}")
    
    # Step 3: Search for relevant memories
    logger.info("Searching for relevant memories")
    search_results = adapter.get_relevant_memories("test memory", test_user_id)
    
    if search_results:
        logger.info(f"Found {len(search_results)} relevant memories")
        for i, memory in enumerate(search_results, 1):
            logger.info(f"Memory {i}: {memory.get('text')} (Score: {memory.get('score')})")
    else:
        logger.warning("No relevant memories found")
    
    # Step 4: Switch modes and test again
    if adapter.current_mode == adapter.MEMORY_MODE_MEM0AI:
        logger.info("Testing fallback mode")
        result = adapter.set_memory_mode(adapter.MEMORY_MODE_FALLBACK)
        logger.info(f"Mode switch result: {result}")
        
        # Add another memory in fallback mode
        test_memory_fallback = f"This is a fallback memory at {time.strftime('%Y-%m-%d %H:%M:%S')}"
        memory_id_fallback = adapter.add_memory(test_memory_fallback, test_user_id)
        logger.info(f"Added fallback memory with ID: {memory_id_fallback}")
        
        # Switch back to auto mode
        adapter.set_memory_mode(adapter.MEMORY_MODE_AUTO)
    
    return True

def test_memory_api():
    """Test memory operations through the API"""
    logger.info("Testing memory operations through API...")
    
    # Set a test user ID
    test_user_id = f"api_test_{int(time.time())}"
    test_memory = f"API test memory at {time.strftime('%Y-%m-%d %H:%M:%S')}"
    
    # Test chat API with memory
    try:
        # Send a chat message
        response = requests.post(
            f"{API_BASE_URL}/chat",
            json={
                "message": "Remember this test message: " + test_memory,
                "model": DEFAULT_MODEL,
                "memory_id": test_user_id,
                "stream": False
            }
        )
        
        if response.status_code == 200:
            logger.info("Successfully sent chat message")
            
            # Wait a bit for memory to be stored
            time.sleep(1)
            
            # Get memories for this user
            response = requests.get(
                f"{API_BASE_URL}/memories",
                params={"memory_id": test_user_id}
            )
            
            if response.status_code == 200:
                data = response.json()
                memories = data.get("memories", [])
                logger.info(f"Found {len(memories)} memories for {test_user_id}")
                
                # Check if our test memory is stored
                found = False
                for memory in memories:
                    if test_memory in memory.get("text", ""):
                        found = True
                        logger.info(f"Found test memory: {memory.get('text')[:50]}...")
                
                if not found:
                    logger.warning("Test memory not found in API results")
                
                return found
            else:
                logger.error(f"Failed to get memories: {response.status_code} {response.text}")
                return False
        else:
            logger.error(f"Failed to send chat message: {response.status_code} {response.text}")
            return False
    except Exception as e:
        logger.error(f"Error in memory API test: {e}")
        return False

def run_all_tests():
    """Run all tests and return overall success"""
    print("\n=== Vector Integration Test ===\n")
    
    # Test 1: Service availability
    print("\n--- Testing Service Availability ---")
    services_ok = test_services_availability()
    
    if not services_ok:
        print("\n❌ Services test failed. Cannot continue.")
        return False
    
    print("\n✅ Services test passed.")
    
    # Test 2: API endpoints
    print("\n--- Testing API Endpoints ---")
    try:
        api_ok = test_api_endpoints()
        if api_ok:
            print("\n✅ API endpoints test passed.")
        else:
            print("\n❌ API endpoints test failed.")
    except Exception as e:
        print(f"\n❌ API endpoints test error: {e}")
        api_ok = False
    
    # Test 3: Direct memory operations
    print("\n--- Testing Direct Memory Operations ---")
    memory_ok = test_memory_operations()
    
    if memory_ok:
        print("\n✅ Memory operations test passed.")
    else:
        print("\n❌ Memory operations test failed.")
    
    # Test 4: Memory API
    print("\n--- Testing Memory API ---")
    memory_api_ok = test_memory_api()
    
    if memory_api_ok:
        print("\n✅ Memory API test passed.")
    else:
        print("\n❌ Memory API test failed.")
    
    # Overall results
    overall = services_ok and memory_ok
    
    if overall:
        print("\n✅ All critical tests passed!")
    else:
        print("\n❌ Some tests failed.")
    
    return overall

if __name__ == "__main__":
    # Get API URL from command line if provided
    if len(sys.argv) > 1:
        API_BASE_URL = sys.argv[1]
        
    # Run the tests
    success = run_all_tests()
    sys.exit(0 if success else 1)
