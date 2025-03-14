"""
Example chatbot application using the mem0ai adapter

This simple command-line application demonstrates how to use the mem0ai adapter
to create a chatbot with memory capabilities.
"""

import os
import sys
import logging
import time
import requests
import json
from typing import List, Dict, Any, Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import the mem0ai adapter
try:
    from final_mem0_adapter import Mem0Adapter
    logger.info("Successfully imported Mem0Adapter")
except ImportError as e:
    logger.error(f"Failed to import Mem0Adapter: {e}")
    sys.exit(1)

# Configuration
OLLAMA_URL = "http://localhost:11434"
DEFAULT_MODEL = "llama3"
USER_ID = "demo_user"
MEMORY_MODE = "auto"  # Options: "auto", "mem0ai", "fallback"

class ChatSession:
    """A simple chat session with memory capabilities"""
    
    def __init__(self, model: str = DEFAULT_MODEL, user_id: str = USER_ID):
        """Initialize the chat session"""
        self.model = model
        self.user_id = user_id
        self.session_id = f"session_{int(time.time())}"
        self.history = []
        
        # Initialize the memory adapter
        try:
            self.memory = Mem0Adapter(
                ollama_host=OLLAMA_URL,
                default_model=model,
                memory_mode=MEMORY_MODE
            )
            logger.info(f"Memory system initialized with mode: {self.memory.memory_mode}")
            logger.info(f"Memory system info: {self.memory.get_memory_system_info()}")
        except Exception as e:
            logger.error(f"Failed to initialize memory: {e}")
            sys.exit(1)
    
    def generate_response(self, user_message: str) -> str:
        """Generate a response to the user message using Ollama API"""
        # Add the user message to history
        self.history.append({"role": "user", "content": user_message})
        
        # Retrieve relevant memories
        memories = self.memory.get_relevant_memories(user_message, self.user_id)
        logger.info(f"Retrieved {len(memories)} relevant memories")
        
        # Prepare system message with memory context
        system_message = "You are a helpful assistant with memory capabilities."
        
        if memories:
            system_message += "\n\nRelevant memories:"
            for i, memory in enumerate(memories, 1):
                system_message += f"\n{i}. {memory['text']}"
        
        # Prepare messages for Ollama API
        messages = [
            {"role": "system", "content": system_message}
        ] + self.history
        
        # Call Ollama API
        try:
            response = requests.post(
                f"{OLLAMA_URL}/api/chat",
                json={
                    "model": self.model,
                    "messages": messages,
                    "stream": False
                },
                timeout=60
            )
            response.raise_for_status()
            result = response.json()
            
            # Extract the assistant response
            assistant_message = result["message"]["content"]
            
            # Add to history
            self.history.append({"role": "assistant", "content": assistant_message})
            
            # Update memory with the conversation
            full_exchange = f"User: {user_message}\nAssistant: {assistant_message}"
            memory_id = self.memory.add_memory(
                full_exchange, 
                self.user_id,
                metadata={
                    "timestamp": time.time(),
                    "session_id": self.session_id
                }
            )
            logger.info(f"Added memory with ID: {memory_id}")
            
            return assistant_message
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return "Sorry, I encountered an error while processing your request."
    
    def list_active_memories(self) -> List[Dict[str, Any]]:
        """List active memories in the current session"""
        active_ids = self.memory.get_active_memories(self.user_id)
        active_memories = []
        
        for memory_id in active_ids:
            memory = self.memory.get_memory(memory_id, self.user_id)
            if memory:
                active_memories.append(memory)
        
        return active_memories
    
    def reset_session(self) -> None:
        """Reset the session history and active memories"""
        self.history = []
        self.memory.reset_active_memories(self.user_id)
        logger.info("Session reset")

def main():
    """Main function to run the chatbot"""
    print("\n=== Memory-Enabled Chatbot ===")
    print("Enter 'quit' to exit, 'reset' to start a new session")
    print("Enter 'memory' to show active memories")
    print("Enter 'status' to show memory system status")
    print("==================================\n")
    
    # Check if Ollama is running
    try:
        response = requests.get(f"{OLLAMA_URL}/api/tags")
        if response.status_code != 200:
            print(f"Error: Ollama server not responding correctly at {OLLAMA_URL}")
            sys.exit(1)
            
        models = response.json().get("models", [])
        available_models = [model["name"] for model in models]
        
        print(f"Available models: {', '.join(available_models)}")
        
        # Use DEFAULT_MODEL or select first available model
        model = DEFAULT_MODEL
        if DEFAULT_MODEL not in available_models and available_models:
            model = available_models[0]
            print(f"Default model not found, using {model} instead")
    except Exception as e:
        print(f"Error: Could not connect to Ollama server: {e}")
        print(f"Please make sure Ollama is running at {OLLAMA_URL}")
        sys.exit(1)
    
    # Create chat session
    session = ChatSession(model=model)
    
    # Main chat loop
    while True:
        try:
            user_input = input("\nYou: ").strip()
            
            if user_input.lower() == 'quit':
                print("Goodbye!")
                break
                
            elif user_input.lower() == 'reset':
                session.reset_session()
                print("Session has been reset.")
                continue
                
            elif user_input.lower() == 'memory':
                active_memories = session.list_active_memories()
                print("\n=== Active Memories ===")
                for i, memory in enumerate(active_memories, 1):
                    print(f"{i}. {memory['text']}")
                print("======================")
                continue
                
            elif user_input.lower() == 'status':
                status = session.memory.get_memory_system_info()
                print("\n=== Memory System Status ===")
                print(f"Mode: {status['mode']}")
                print(f"Current mode: {status['current_mode']}")
                print(f"mem0ai available: {status['mem0ai_available']}")
                print(f"Using fallback: {status['using_fallback']}")
                
                if status['using_fallback']:
                    print(f"Memory file: {status['memory_file']}")
                else:
                    print(f"Vector DB: {status['vector_db']}")
                print("============================")
                continue
                
            elif not user_input:
                continue
            
            # Generate and display response
            response = session.generate_response(user_input)
            print(f"\nAssistant: {response}")
            
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()
