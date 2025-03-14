"""
Simple script to test if mem0ai can be imported correctly
"""

import sys
import logging

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

logger.info(f"Python version: {sys.version}")
logger.info(f"Python executable: {sys.executable}")
logger.info(f"Python path: {sys.path}")

try:
    # Try to import mem0ai
    import mem0ai
    from mem0ai import Memory
    
    logger.info(f"Successfully imported mem0ai version: {mem0ai.__version__ if hasattr(mem0ai, '__version__') else 'unknown'}")
    
    # Try to create a Memory instance with a simple config
    config = {
        "llm": {
            "provider": "ollama",
            "config": {
                "model": "llama3",
                "base_url": "http://localhost:11434"
            }
        },
        "embedder": {
            "provider": "ollama", 
            "config": {
                "model": "nomic-embed-text",
                "base_url": "http://localhost:11434"
            }
        },
        "vector_db": {
            "provider": "in_memory"  # Use in-memory for testing
        }
    }
    
    logger.info("Attempting to create Memory instance...")
    memory = Memory.from_config(config)
    logger.info("Successfully created Memory instance")
    
    # Test a simple memory operation
    test_result = memory.add("Test memory", user_id="test_user")
    logger.info(f"Test add result: {test_result}")
    
except ImportError as e:
    logger.error(f"Failed to import mem0ai: {e}")
    logger.error(f"Import error details: {getattr(e, '__traceback__', 'No traceback')}")
except Exception as e:
    logger.error(f"Error testing mem0ai: {e}")
    logger.error(f"Error type: {type(e)}")
    import traceback
    logger.error(f"Error traceback: {traceback.format_exc()}")
