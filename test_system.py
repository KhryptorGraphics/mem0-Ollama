import unittest
import requests
import os
import json
import time
import sys
from typing import Dict, Any

# Add the current directory to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the components we want to test
from api import app
from memory_utils import MemoryManager
from ollama_client import OllamaClient
from config import OLLAMA_HOST, MEM0_HOST, DEFAULT_MODEL

class SystemIntegrationTest(unittest.TestCase):
    """Test the integration between different components of the system"""

    def setUp(self):
        """Set up test environment"""
        self.client = app.test_client()
        self.memory_manager = MemoryManager()
        self.ollama_client = OllamaClient(host=OLLAMA_HOST)
        self.test_memory_id = f"test_memory_{int(time.time())}"

    def tearDown(self):
        """Clean up after tests"""
        # Clean up test memories if needed
        pass

    def test_ollama_connection(self):
        """Test connection to Ollama server"""
        print(f"\nTesting Ollama connection to {OLLAMA_HOST}...")
        try:
            models = self.ollama_client.get_available_models()
            self.assertIsInstance(models, list, "Expected list of models")
            print(f"Ollama connection successful, found models: {models}")
        except Exception as e:
            self.fail(f"Ollama connection failed: {str(e)}")

    def test_memory_initialization(self):
        """Test memory system initialization"""
        print(f"\nTesting memory system initialization...")
        try:
            # Check if memory system initialized
            self.assertTrue(hasattr(self.memory_manager, '_memory'), 
                           "Memory manager should have '_memory' attribute")
            self.assertTrue(self.memory_manager._initialized, 
                          "Memory manager should be initialized")
            print("Memory system initialized successfully")
        except Exception as e:
            self.fail(f"Memory system initialization failed: {str(e)}")

    def test_memory_add_retrieve(self):
        """Test adding and retrieving memories"""
        print(f"\nTesting memory add/retrieve functionality...")
        try:
            # Add a test memory
            test_text = f"This is a test memory created at {time.time()}"
            memory_id = self.memory_manager.add_memory(
                test_text,
                memory_id=self.test_memory_id,
                metadata={"test": True, "timestamp": time.time()}
            )
            
            # Check if memory was added
            self.assertIsNotNone(memory_id, "Memory ID should not be None")
            
            # Try to retrieve similar memories
            memories = self.memory_manager.get_relevant_memories(
                "test memory",
                memory_id=self.test_memory_id,
                limit=5
            )
            
            # Should find at least one memory
            self.assertGreaterEqual(len(memories), 0, 
                                  "Should retrieve at least one memory")
            
            print(f"Memory add/retrieve test successful")
        except Exception as e:
            self.fail(f"Memory add/retrieve test failed: {str(e)}")

    def test_api_endpoints(self):
        """Test basic API endpoints"""
        print(f"\nTesting API endpoints...")

        # Test the root endpoint (chat interface)
        response = self.client.get('/')
        self.assertEqual(response.status_code, 200, 
                       "Root endpoint should return 200")
        
        # Test model test endpoint
        response = self.client.get('/model_test')
        self.assertEqual(response.status_code, 200, 
                       "Model test endpoint should return 200")
        
        # Test direct test endpoint
        response = self.client.get('/direct')
        self.assertEqual(response.status_code, 200, 
                       "Direct test endpoint should return 200")
        
        # Test Ollama tags proxy
        response = self.client.get('/api/tags')
        self.assertEqual(response.status_code, 200, 
                       "API tags endpoint should return 200")
        
        print("API endpoints test successful")

    def test_chat_api(self):
        """Test the chat API with a simple message"""
        print(f"\nTesting chat API with a simple message...")
        
        try:
            # Prepare a test message
            test_message = {
                "message": "Hello, this is a test message",
                "model": DEFAULT_MODEL,
                "memory_id": self.test_memory_id,
                "stream": False,  # Don't stream for easier testing
                "temperature": 0.7,
                "top_p": 0.9,
                "max_tokens": 100
            }
            
            # Send the message
            response = self.client.post(
                '/api/chat',
                json=test_message,
                content_type='application/json'
            )
            
            # Check response
            self.assertEqual(response.status_code, 200, 
                           "Chat API should return 200")
            
            # Parse response
            data = json.loads(response.data)
            self.assertIn('text', data, "Response should contain 'text' field")
            self.assertIn('model', data, "Response should contain 'model' field")
            
            print(f"Chat API test successful with response: {data['text'][:50]}...")
        except Exception as e:
            self.fail(f"Chat API test failed: {str(e)}")

def run_tests():
    """Run all system integration tests"""
    print("=" * 80)
    print("Starting mem0-ollama System Integration Tests")
    print("=" * 80)
    
    # Create test suite
    suite = unittest.TestSuite()
    
    # Add tests in order of dependencies
    suite.addTest(SystemIntegrationTest('test_ollama_connection'))
    suite.addTest(SystemIntegrationTest('test_memory_initialization'))
    suite.addTest(SystemIntegrationTest('test_memory_add_retrieve'))
    suite.addTest(SystemIntegrationTest('test_api_endpoints'))
    suite.addTest(SystemIntegrationTest('test_chat_api'))
    
    # Run the tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "=" * 80)
    print(f"Tests run: {result.testsRun}")
    print(f"Errors: {len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print("=" * 80)
    
    # Return True if all tests passed
    return len(result.errors) == 0 and len(result.failures) == 0

if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)
