"""
Test the final improved mem0ai adapter
"""

import os
import sys
import logging
import time

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Try to import the adapter
try:
    from final_mem0_adapter import Mem0Adapter
    logger.info("Successfully imported Mem0Adapter from final implementation")
except ImportError as e:
    logger.error(f"Failed to import Mem0Adapter: {e}")
    sys.exit(1)

def test_adapter_initialization():
    """Test that the adapter can be initialized"""
    try:
        # Create an instance with default settings
        adapter = Mem0Adapter()
        logger.info(f"Created adapter with mode: {adapter.memory_mode}")
        
        # Get system info
        info = adapter.get_memory_system_info()
        logger.info(f"Memory system info: {info}")
        
        return adapter
    except Exception as e:
        logger.error(f"Error initializing adapter: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return None

def test_memory_operations(adapter):
    """Test basic memory operations"""
    if not adapter:
        logger.error("Adapter not available for testing")
        return False
    
    # First make sure we're in fallback mode as it's more reliable for testing
    logger.info("Switching to fallback mode for reliable testing")
    result = adapter.set_memory_mode(adapter.MEMORY_MODE_FALLBACK)
    logger.info(f"Mode switching result: {result}")
    
    if adapter.current_mode != adapter.MEMORY_MODE_FALLBACK:
        logger.error(f"Failed to switch to fallback mode, current mode is {adapter.current_mode}")
        return False
    
    try:
        # Test adding a memory
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        test_text = f"This is a test memory created at {timestamp}"
        memory_id = adapter.add_memory(test_text, "test_user")
        logger.info(f"Added memory with ID: {memory_id}")
        
        if not memory_id:
            logger.error("Failed to add memory")
            return False
        
        # Test retrieving relevant memories
        memories = adapter.get_relevant_memories("test memory", "test_user")
        if memories:
            logger.info(f"Retrieved {len(memories)} memories")
            for memory in memories:
                logger.info(f"Memory: {memory}")
        else:
            logger.warning("No memories retrieved")
        
        # Test listing all memories
        all_memories = adapter.list_memories("test_user")
        logger.info(f"Listed {len(all_memories)} memories for test_user")
        
        # Test active memories tracking
        active_ids = adapter.get_active_memories("test_user")
        logger.info(f"Active memory IDs: {active_ids}")
        
        # Reset active memories
        adapter.reset_active_memories("test_user")
        active_ids = adapter.get_active_memories("test_user")
        logger.info(f"Active memory IDs after reset: {active_ids}")
        
        return True
    except Exception as e:
        logger.error(f"Error in memory operations: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False

def test_mode_switching(adapter):
    """Test switching between memory modes"""
    if not adapter:
        logger.error("Adapter not available for testing")
        return False
    
    try:
        # Get current mode
        current_mode = adapter.current_mode
        logger.info(f"Current mode before switching: {current_mode}")
        
        # Try to switch to fallback mode
        result = adapter.set_memory_mode(adapter.MEMORY_MODE_FALLBACK)
        logger.info(f"Mode switching result: {result}")
        logger.info(f"Current mode after switching: {adapter.current_mode}")
        
        # Try to add a memory in fallback mode
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        test_text = f"This is a fallback mode memory created at {timestamp}"
        memory_id = adapter.add_memory(test_text, "test_user")
        logger.info(f"Added memory in fallback mode with ID: {memory_id}")
        
        # Switch back to auto mode
        result = adapter.set_memory_mode(adapter.MEMORY_MODE_AUTO)
        logger.info(f"Mode switching back result: {result}")
        logger.info(f"Final mode: {adapter.current_mode}")
        
        return True
    except Exception as e:
        logger.error(f"Error in mode switching: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False

def run_all_tests():
    """Run all tests"""
    logger.info("Starting adapter tests...")
    
    # Test initialization
    adapter = test_adapter_initialization()
    if not adapter:
        logger.error("Initialization test failed")
        return False
    
    # Test memory operations
    if not test_memory_operations(adapter):
        logger.error("Memory operations test failed")
        return False
    
    # Test mode switching
    if not test_mode_switching(adapter):
        logger.error("Mode switching test failed")
        return False
    
    logger.info("All tests completed successfully!")
    return True

if __name__ == "__main__":
    success = run_all_tests()
    logger.info(f"Tests {'succeeded' if success else 'failed'}")
    sys.exit(0 if success else 1)
