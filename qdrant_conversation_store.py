"""
Qdrant MCP conversation storage module for mem0-ollama

This module provides integration with Qdrant MCP for storing all conversations
in a vector database, separate from the existing memory system.
"""

import logging
import time
import json
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple

# Configure logging
logger = logging.getLogger(__name__)

class QdrantConversationStore:
    """
    Store conversations in Qdrant using the MCP tool.
    This class provides methods to store user and assistant messages with proper formatting
    and metadata for effective retrieval.
    """
    
    def __init__(self, enabled: bool = True):
        """
        Initialize the Qdrant conversation store
        
        Args:
            enabled: Whether to enable conversation storage in Qdrant
        """
        self.enabled = enabled
        self.storage_initialized = False
        self.last_error = None
        logger.info(f"Qdrant MCP conversation store initialized (enabled: {enabled})")
    
    def store_conversation(
        self, 
        user_message: str, 
        assistant_response: str, 
        conversation_id: str = "default",
        metadata: Optional[Dict[str, Any]] = None
    ) -> Tuple[bool, Optional[str]]:
        """
        Store a conversation exchange in Qdrant MCP
        
        Args:
            user_message: The user's message
            assistant_response: The assistant's response
            conversation_id: Identifier for the conversation
            metadata: Additional metadata to store
            
        Returns:
            Tuple of (success, error_message)
        """
        if not self.enabled:
            return (False, "Storage is disabled")
        
        try:
            # Format conversation text with proper spacing and structure
            formatted_text = self._format_conversation_text(user_message, assistant_response)
            
            # Prepare metadata
            timestamp = time.time()
            formatted_time = datetime.fromtimestamp(timestamp).isoformat()
            
            meta = {
                "timestamp": timestamp,
                "formatted_time": formatted_time,
                "conversation_id": conversation_id,
                "type": "conversation",
                "user_message_length": len(user_message),
                "assistant_response_length": len(assistant_response)
            }
            
            # Merge with provided metadata
            if metadata:
                meta.update(metadata)
            
            # Use the MCP tool to store the conversation
            from importlib import import_module
            try:
                # Import the special MCP module that provides the qdrant tool
                qdrant_mcp = import_module("qdrant_mcp")
                
                # Store the conversation in Qdrant using the MCP tool
                result = qdrant_mcp.store(
                    information=formatted_text,
                    metadata=meta
                )
                
                # Log the result
                logger.info(f"Stored conversation in Qdrant MCP: {result}")
                return (True, None)
                
            except ImportError:
                self.last_error = "Qdrant MCP module not available"
                logger.error(f"Cannot import qdrant_mcp module: {self.last_error}")
                return (False, self.last_error)
                
            except Exception as e:
                self.last_error = f"Error storing in Qdrant MCP: {str(e)}"
                logger.error(self.last_error)
                return (False, self.last_error)
                
        except Exception as e:
            self.last_error = f"Unexpected error in store_conversation: {str(e)}"
            logger.error(self.last_error)
            return (False, self.last_error)
    
    def find_conversations(
        self, 
        query: str, 
        limit: int = 5
    ) -> Tuple[bool, Optional[List[Dict[str, Any]]], Optional[str]]:
        """
        Find conversations in Qdrant MCP based on a query
        
        Args:
            query: The search query
            limit: Maximum number of results to return
            
        Returns:
            Tuple of (success, results, error_message)
        """
        if not self.enabled:
            return (False, None, "Storage is disabled")
        
        try:
            from importlib import import_module
            try:
                # Import the special MCP module that provides the qdrant tool
                qdrant_mcp = import_module("qdrant_mcp")
                
                # Find conversations in Qdrant using the MCP tool
                result = qdrant_mcp.find(query=query)
                
                # Process results
                if result and isinstance(result, dict) and "results" in result:
                    # Format and return results
                    formatted_results = []
                    for item in result["results"][:limit]:
                        formatted_results.append({
                            "text": item.get("text", ""),
                            "score": item.get("score", 0),
                            "metadata": item.get("metadata", {})
                        })
                    
                    return (True, formatted_results, None)
                else:
                    return (True, [], None)  # No results
                
            except ImportError:
                self.last_error = "Qdrant MCP module not available"
                logger.error(f"Cannot import qdrant_mcp module: {self.last_error}")
                return (False, None, self.last_error)
                
            except Exception as e:
                self.last_error = f"Error searching in Qdrant MCP: {str(e)}"
                logger.error(self.last_error)
                return (False, None, self.last_error)
                
        except Exception as e:
            self.last_error = f"Unexpected error in find_conversations: {str(e)}"
            logger.error(self.last_error)
            return (False, None, self.last_error)
    
    def _format_conversation_text(self, user_message: str, assistant_response: str) -> str:
        """
        Format the conversation text with proper spacing and structure
        
        Args:
            user_message: The user's message
            assistant_response: The assistant's response
            
        Returns:
            Formatted conversation text
        """
        # Ensure proper spacing and structure
        formatted_text = f"User: {user_message.strip()}\n\nAssistant: {assistant_response.strip()}"
        
        # Extract any thinking section for separate storage
        thinking_match = self._extract_thinking(assistant_response)
        if thinking_match:
            thinking_content = thinking_match.strip()
            formatted_text += f"\n\nThinking Process:\n{thinking_content}"
        
        return formatted_text
    
    def _extract_thinking(self, text: str) -> Optional[str]:
        """
        Extract thinking sections from text
        
        Args:
            text: The text to extract thinking from
            
        Returns:
            The extracted thinking text or None
        """
        import re
        # Match both <think> and <thinking> tags
        thinking_pattern = re.compile(r'<think(?:ing)?>(.+?)</think(?:ing)?>', re.DOTALL)
        
        match = thinking_pattern.search(text)
        if match:
            return match.group(1)
        return None

# Initialize the global conversation store
conversation_store = QdrantConversationStore()
