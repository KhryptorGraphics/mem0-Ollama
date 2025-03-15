"""
Test script for verifying Qdrant MCP integration

This script tests the integration with the Qdrant MCP tool for storing and retrieving conversations.
"""

import os
import sys
import logging
import json
import time
from typing import Dict, Any, List, Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import local modules
from config import QDRANT_MCP_ENABLED
from qdrant_conversation_store import conversation_store

# Test data
TEST_USER_MESSAGE = "Hello, this is a test message for Qdrant MCP storage."
TEST_ASSISTANT_RESPONSE = "Hi there! I'm responding to your test message. This will be stored in Qdrant MCP."
TEST_CONVERSATION_ID = f"test_conversation_{int(time.time())}"

def test_qdrant_mcp_enabled():
    """Test that Qdrant MCP is enabled in the configuration"""
    logger.info("Testing Qdrant MCP configuration...")
    
    if not QDRANT_MCP_ENABLED:
        logger.error("❌ Qdrant MCP is not enabled in configuration")
        return False
    
    logger.info("✅ Qdrant MCP is enabled in configuration")
    return True

def test_store_conversation():
    """Test storing a conversation in Qdrant MCP"""
    logger.info("Testing conversation storage in Qdrant MCP...")
    
    try:
        # Store a test conversation
        success, error = conversation_store.store_conversation(
            user_message=TEST_USER_MESSAGE,
            assistant_response=TEST_ASSISTANT_RESPONSE,
            conversation_id=TEST_CONVERSATION_ID,
            metadata={
                'test': True,
                'timestamp': time.time()
            }
        )
        
        if not success:
            logger.error(f"❌ Failed to store conversation in Qdrant MCP: {error}")
            return False
        
        logger.info("✅ Successfully stored conversation in Qdrant MCP")
        return True
    except Exception as e:
        logger.error(f"❌ Exception while storing conversation: {e}")
        return False

def test_find_conversations():
    """Test finding conversations in Qdrant MCP"""
    logger.info("Testing conversation retrieval from Qdrant MCP...")
    
    try:
        # Wait a moment for indexing to complete
        time.sleep(1)
        
        # Search for the test conversation
        success, results, error = conversation_store.find_conversations(
            query="test message",
            limit=5
        )
        
        if not success:
            logger.error(f"❌ Failed to search conversations in Qdrant MCP: {error}")
            return False
        
        if not results or len(results) == 0:
            logger.warning("⚠️ No results found when searching for test message")
            return False
        
        # Check if our test conversation is in the results
        found = False
        for result in results:
            if TEST_USER_MESSAGE in result.get('text', ''):
                found = True
                logger.info(f"Found test conversation with score: {result.get('score', 0)}")
                break
        
        if not found:
            logger.warning("⚠️ Test conversation not found in search results")
            return False
        
        logger.info("✅ Successfully retrieved conversation from Qdrant MCP")
        return True
    except Exception as e:
        logger.error(f"❌ Exception while searching conversations: {e}")
        return False

def test_thinking_extraction():
    """Test extracting thinking content from responses"""
    logger.info("Testing thinking content extraction...")
    
    try:
        import re
        
        # Define the same pattern used in api.py
        thinking_pattern = re.compile(r'<think(?:ing)?>(.+?)</think(?:ing)?>', re.DOTALL)
        
        # Local implementation of extract_thinking_content
        def extract_thinking(text):
            thinking_content = []
            for match in thinking_pattern.finditer(text):
                if match.group(1):
                    thinking_content.append(match.group(1).strip())
            return thinking_content
        
        # Local implementation of remove_thinking_tags
        def remove_thinking(text):
            return thinking_pattern.sub('', text).strip()
        
        # Test text with thinking tags
        test_text = "Hello! <thinking>This is my thinking process</thinking> Here's my response."
        
        # Test extraction
        thinking_content = extract_thinking(test_text)
        if not thinking_content or len(thinking_content) == 0:
            logger.error("❌ Failed to extract thinking content")
            return False
        
        logger.info(f"Extracted thinking content: {thinking_content}")
        
        # Test removal
        cleaned_text = remove_thinking(test_text)
        if "<thinking>" in cleaned_text or "</thinking>" in cleaned_text:
            logger.error("❌ Failed to remove thinking tags")
            return False
        
        logger.info(f"Cleaned text: {cleaned_text}")
        logger.info("✅ Successfully extracted and removed thinking content")
        return True
    except Exception as e:
        logger.error(f"❌ Exception while testing thinking extraction: {e}")
        return False

def run_all_tests():
    """Run all tests and return overall success"""
    print("\n=== Qdrant MCP Integration Test ===\n")
    
    # Test 1: Configuration
    print("\n--- Testing Configuration ---")
    config_ok = test_qdrant_mcp_enabled()
    
    if not config_ok:
        print("\n❌ Configuration test failed. Cannot continue.")
        return False
    
    print("\n✅ Configuration test passed.")
    
    # Test 2: Storage
    print("\n--- Testing Storage ---")
    storage_ok = test_store_conversation()
    
    if not storage_ok:
        print("\n❌ Storage test failed. Cannot continue.")
        return False
    
    print("\n✅ Storage test passed.")
    
    # Test 3: Retrieval
    print("\n--- Testing Retrieval ---")
    retrieval_ok = test_find_conversations()
    
    if retrieval_ok:
        print("\n✅ Retrieval test passed.")
    else:
        print("\n❌ Retrieval test failed.")
    
    # Test 4: Thinking extraction
    print("\n--- Testing Thinking Extraction ---")
    thinking_ok = test_thinking_extraction()
    
    if thinking_ok:
        print("\n✅ Thinking extraction test passed.")
    else:
        print("\n❌ Thinking extraction test failed.")
    
    # Overall results
    overall = config_ok and storage_ok and retrieval_ok and thinking_ok
    
    if overall:
        print("\n✅ All tests passed!")
    else:
        print("\n⚠️ Some tests failed.")
    
    return overall

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
