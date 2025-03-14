"""
Test script for the new mem0 adapter with better import handling
"""

import logging
import sys
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Print system info
logger.info(f"Python version: {sys.version}")
logger.info(f"Python executable: {sys.executable}")
logger.info(f"Current working directory: {os.getcwd()}")
logger.info(f"Python path: {sys.path}")

try:
    # Try to import the new adapter
    from new_mem0_adapter import Mem0Adapter
    
    logger.info("Successfully imported Mem0Adapter from new_mem0_adapter.py")
    
    # Create an adapter instance
    adapter = Mem0Adapter()
    logger.info(f"Created Mem0Adapter instance with mode: {adapter.memory_mode}")
    logger.info(f"Current memory system info: {adapter.get_memory_system_info()}")
    
    # Test adding a memory
    test_text = "This is a test memory created at " + str(os.environ.get("TEST_ENV_VAR", "runtime"))
    memory_id = adapter.add_memory(test_text, "test_user")
    logger.info(f"Added memory with ID: {memory_id}")
    
    # Test retrieving memories
    logger.info("Testing memory retrieval...")
    memories = adapter.get_relevant_memories("test memory", "test_user")
    
    if memories:
        logger.info(f"Retrieved {len(memories)} memories")
        for memory in memories:
            logger.info(f"Memory: {memory}")
    else:
        logger.info("No memories retrieved")
    
    # Test listing memories
    all_memories = adapter.list_memories("test_user")
    logger.info(f"Listed {len(all_memories)} memories for test_user")
    
except ImportError as e:
    logger.error(f"Failed to import Mem0Adapter: {e}")
except Exception as e:
    logger.error(f"Error testing Mem0Adapter: {e}")
    import traceback
    logger.error(f"Error traceback: {traceback.format_exc()}")
