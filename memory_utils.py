"""
Memory management utilities for mem0 + Ollama integration

This module handles memory operations including storage, retrieval, and management
of conversation history using Qdrant vector database.
"""

import time
import requests
import json
from typing import Dict, List, Any, Optional, Union, Tuple

from mem0 import Memory
from config import (
    OLLAMA_HOST, 
    OLLAMA_MODEL, 
    QDRANT_HOST, 
    QDRANT_COLLECTION,
    MODEL_DIMENSIONS
)

import logging_utils

# Set up loggers
logger = logging_utils.get_logger(__name__)
memory_logger = logging_utils.get_logger("memory")

# Default retry settings
DEFAULT_TIMEOUT = 10  # seconds
MAX_RETRIES = 3
RETRY_DELAY = 1.0  # seconds

def preprocess_user_message(message: str) -> str:
    """
    Preprocess user message to enhance it for better memory storage and retrieval.
    
    This function makes user messages more prominent in the vector store by:
    1. Adding emphasis markers
    2. Potentially repeating key phrases
    3. Formatting for better embedding
    
    Args:
        message: The original user message
        
    Returns:
        Enhanced version of the message optimized for embedding and retrieval
    """
    logger.debug(f"Preprocessing user message (length: {len(message)})")
    
    # Skip preprocessing for very short messages
    if len(message) < 5:
        enhanced = f"IMPORTANT USER QUERY: {message}"
        logger.debug("Applied short message enhancement")
        return enhanced
        
    # For longer messages, enhance with emphasis and formatting
    enhanced = message.strip()
    
    # Add importance markers at the beginning and end
    enhanced = f"IMPORTANT USER INPUT: {enhanced} [USER QUERY END]"
    
    # If message is a question, emphasize it further
    if any(q in message for q in ["?", "what", "how", "why", "when", "where", "who", "which"]):
        enhanced = f"USER QUESTION: {enhanced}"
        logger.debug("Applied question enhancement")
    
    logger.debug(f"Message preprocessing complete (original length: {len(message)}, enhanced length: {len(enhanced)})")
    return enhanced

@logging_utils.timed_operation(operation="check_qdrant")
def check_qdrant(timeout: int = DEFAULT_TIMEOUT, retries: int = MAX_RETRIES) -> Tuple[bool, str]:
    """
    Check if Qdrant is running and accessible.
    
    Args:
        timeout: Request timeout in seconds
        retries: Maximum number of retry attempts
        
    Returns:
        Tuple of (success, message)
    """
    logger.info(f"Checking Qdrant connection at {QDRANT_HOST}")
    request_id = logging_utils.set_request_id()
    
    # Try dashboard first, then collections endpoint
    endpoints = ["/dashboard/", "/collections"]
    
    for endpoint in endpoints:
        url = f"{QDRANT_HOST}{endpoint}"
        attempt = 0
        
        while attempt <= retries:
            attempt += 1
            
            try:
                start_time = time.time()
                
                # Log the API call
                logging_utils.log_api_call(
                    memory_logger,
                    method="GET",
                    url=url
                )
                
                # Make the request
                response = requests.get(url, timeout=timeout)
                duration_ms = (time.time() - start_time) * 1000
                
                # Log response
                logging_utils.log_api_call(
                    memory_logger,
                    method="GET",
                    url=url,
                    response=response.text[:200] if response.status_code != 200 else "OK",
                    status_code=response.status_code,
                    duration_ms=duration_ms
                )
                
                if response.status_code == 200:
                    success_msg = f"Qdrant is running at {QDRANT_HOST}"
                    logger.info(success_msg)
                    return True, success_msg
                else:
                    logger.warning(f"Qdrant endpoint {endpoint} returned status {response.status_code}")
                    
                    # Try next endpoint if this one failed
                    break
                    
            except requests.Timeout as e:
                error_msg = f"Qdrant connection timed out after {timeout}s"
                logging_utils.log_api_call(
                    memory_logger,
                    method="GET",
                    url=url,
                    error=e,
                    status_code=408
                )
                
                if attempt <= retries:
                    backoff = RETRY_DELAY * (2 ** (attempt - 1))
                    logger.warning(f"Timeout. Retrying after {backoff:.2f}s (attempt {attempt}/{retries})")
                    time.sleep(backoff)
                    continue
                else:
                    # Move to the next endpoint if all retries failed
                    break
                    
            except requests.RequestException as e:
                error_msg = f"Error connecting to Qdrant: {str(e)}"
                logging_utils.log_api_call(
                    memory_logger,
                    method="GET",
                    url=url,
                    error=e,
                    status_code=500
                )
                
                if attempt <= retries:
                    backoff = RETRY_DELAY * (2 ** (attempt - 1))
                    logger.warning(f"Connection error. Retrying after {backoff:.2f}s (attempt {attempt}/{retries})")
                    time.sleep(backoff)
                    continue
                else:
                    # Move to the next endpoint if all retries failed
                    break
    
    # If we get here, all endpoints failed
    error_msg = f"Qdrant is not running or not accessible at {QDRANT_HOST}"
    logger.error(error_msg)
    return False, error_msg

@logging_utils.timed_operation(operation="initialize_memory")
def initialize_memory(
    ollama_model: str = OLLAMA_MODEL,
    embed_model: Optional[str] = None,
    unified_memory: bool = True
) -> Memory:
    """
    Initialize the Memory object with Ollama and Qdrant configurations.
    
    Args:
        ollama_model: Ollama model to use for LLM
        embed_model: Ollama model to use for embeddings (defaults to same as ollama_model)
        unified_memory: Whether to use unified memory for all models
    
    Returns:
        Memory object configured with Ollama and Qdrant
    """
    logger.info(f"Initializing Memory with Ollama ({ollama_model}) and Qdrant ({QDRANT_HOST})")
    
    # Set request ID for this operation
    request_id = logging_utils.set_request_id()
    
    if embed_model is None:
        # For embedding, prefer specialized embedding models if available
        try:
            logger.info("Checking for specialized embedding models")
            start_time = time.time()
            
            embed_check_url = f"{OLLAMA_HOST}/api/tags"
            
            # Log API call
            logging_utils.log_api_call(
                memory_logger,
                method="GET",
                url=embed_check_url
            )
            
            embed_check = requests.get(embed_check_url, timeout=10)
            duration_ms = (time.time() - start_time) * 1000
            
            # Log response
            if embed_check.status_code == 200:
                available_models = [m.get("name") for m in embed_check.json().get("models", [])]
                logging_utils.log_api_call(
                    memory_logger,
                    method="GET",
                    url=embed_check_url,
                    response=f"Found {len(available_models)} models",
                    status_code=embed_check.status_code,
                    duration_ms=duration_ms
                )
                
                logger.info(f"Available models for embeddings: {', '.join(available_models[:5])}")
                
                if "nomic-embed-text" in available_models or "nomic-embed-text:latest" in available_models:
                    embed_model = "nomic-embed-text"
                    logger.info("Using nomic-embed-text model for embeddings")
                elif "snowflake-arctic-embed" in available_models or "snowflake-arctic-embed:latest" in available_models:
                    embed_model = "snowflake-arctic-embed"
                    logger.info("Using snowflake-arctic-embed model for embeddings")
                else:
                    embed_model = ollama_model
                    logger.info(f"Using {ollama_model} for embeddings (specialized embedding models not found)")
            else:
                # Log error response
                logging_utils.log_api_call(
                    memory_logger,
                    method="GET",
                    url=embed_check_url,
                    response=embed_check.text,
                    status_code=embed_check.status_code,
                    duration_ms=duration_ms
                )
                
                logger.warning(f"Failed to check for embedding models: {embed_check.status_code}")
                embed_model = ollama_model
                logger.info(f"Defaulting to {ollama_model} for embeddings")
                
        except Exception as e:
            logger.error(f"Error checking for embedding models: {e}")
            logging_utils.log_exception(logger, e, {"url": f"{OLLAMA_HOST}/api/tags"})
            embed_model = ollama_model
            logger.info(f"Using {ollama_model} for embeddings after error")
    
    # Determine embedding dimensions based on model
    model_key = embed_model.split(':')[0]
    embed_dims = MODEL_DIMENSIONS.get(model_key, 768)
    logger.info(f"Using embedding dimensions: {embed_dims} for model {embed_model}")
    
    config = {
        "vector_store": {
            "provider": "qdrant",
            "config": {
                "collection_name": QDRANT_COLLECTION,
                "host": QDRANT_HOST.replace("http://", "").replace("https://", "").split(":")[0],
                "port": int(QDRANT_HOST.split(":")[-1]) if ":" in QDRANT_HOST else 6333,
                "embedding_model_dims": embed_dims,
                # "unified_memory" is not a supported field, removed
            },
        },
        "llm": {
            "provider": "ollama",
            "config": {
                "model": ollama_model,
                "temperature": 0.7,
                "max_tokens": 2000,
                "ollama_base_url": OLLAMA_HOST,
            },
        },
        "embedder": {
            "provider": "ollama",
            "config": {
                "model": embed_model,
                "ollama_base_url": OLLAMA_HOST,
                "embedding_dims": embed_dims,
            },
        },
    }
    
    logger.info(f"Memory configuration prepared: {json.dumps(config, indent=2)}")
    
    try:
        logger.info("Creating Memory instance from config")
        memory = Memory.from_config(config)
        logger.info("Memory initialization successful")
        
        # Initialize the memory status tracker after creating memory
        initialize_memory_status_tracking()
        return memory
    except Exception as e:
        logger.error(f"Error initializing Memory: {e}")
        logging_utils.log_exception(logger, e, {"config": config})
        raise RuntimeError(f"Failed to initialize memory system: {str(e)}") from e

# Global constants
GLOBAL_MEMORY_ID = "global_memory_store"
MEMORY_COUNTER = {"active": 0, "inactive": 0, "total": 0}

# Key for storing memory status information in Qdrant
STATUS_KEY = "memory_status.json"

@logging_utils.timed_operation(operation="initialize_memory_status_tracking")
def initialize_memory_status_tracking():
    """Initialize the memory status tracking, loading any existing data."""
    logger.info("Initializing memory status tracking")
    request_id = logging_utils.set_request_id()
    
    try:
        # Try to load existing status from Qdrant
        url = f"{QDRANT_HOST}/collections/{QDRANT_COLLECTION}/points/scroll"
        payload = {"limit": 1000, "with_payload": True}
        
        start_time = time.time()
        
        # Log API call
        logging_utils.log_api_call(
            memory_logger,
            method="POST",
            url=url,
            payload=payload
        )
        
        response = requests.post(url, json=payload, timeout=DEFAULT_TIMEOUT)
        duration_ms = (time.time() - start_time) * 1000
        
        # Log response
        success = response.status_code == 200
        if success:
            data = response.json()
            point_count = len(data.get("result", []))
            logging_utils.log_api_call(
                memory_logger,
                method="POST",
                url=url,
                response=f"Retrieved {point_count} points",
                status_code=response.status_code,
                duration_ms=duration_ms
            )
        else:
            logging_utils.log_api_call(
                memory_logger,
                method="POST",
                url=url,
                response=response.text,
                status_code=response.status_code,
                error=Exception(f"Failed to retrieve memory points: {response.status_code}"),
                duration_ms=duration_ms
            )
            
            logger.warning(f"Failed to retrieve memory points: {response.status_code}, {response.text[:200]}")
            return
            
        if response.status_code == 200:
            data = response.json()
            active_count = 0
            inactive_count = 0
            
            # Count active and inactive memories
            for point in data.get("result", []):
                payload = point.get("payload", {})
                # Check if the memory is inactive
                is_inactive = payload.get("inactive", False)
                if is_inactive:
                    inactive_count += 1
                else:
                    active_count += 1
            
            # Update the global counter
            global MEMORY_COUNTER
            MEMORY_COUNTER["active"] = active_count
            MEMORY_COUNTER["inactive"] = inactive_count
            MEMORY_COUNTER["total"] = active_count + inactive_count
            
            logger.info(f"Memory status initialized: {MEMORY_COUNTER}")
            
            # Log memory operation
            logging_utils.log_memory_operation(
                memory_logger,
                operation="initialize",
                user_id=GLOBAL_MEMORY_ID,
                success=True,
                details={
                    "active": active_count,
                    "inactive": inactive_count,
                    "total": active_count + inactive_count
                },
                duration_ms=duration_ms
            )
        else:
            logger.warning(f"Failed to initialize memory status. Using default values.")
            logging_utils.log_memory_operation(
                memory_logger,
                operation="initialize",
                user_id=GLOBAL_MEMORY_ID,
                success=False,
                error=Exception(f"Failed with status code: {response.status_code}")
            )
    except Exception as e:
        logger.error(f"Error initializing memory status tracking: {e}")
        logging_utils.log_exception(logger, e, {"collection": QDRANT_COLLECTION})
        logging_utils.log_memory_operation(
            memory_logger,
            operation="initialize",
            user_id=GLOBAL_MEMORY_ID,
            success=False,
            error=e
        )

@logging_utils.timed_operation(operation="chat_with_memories")
def chat_with_memories(
    memory: Memory, 
    message: str, 
    user_id: str = "default_user",  # This parameter is kept for API compatibility but ignored
    memory_mode: str = "search",    # This parameter is kept for API compatibility but ignored
    output_format: Optional[Union[str, Dict]] = None,
    model: Optional[str] = None,
    temperature: float = 0.7,       # Added temperature parameter
    max_tokens: int = 2000          # Added max tokens parameter
) -> Dict[str, Any]:
    """
    Process a chat message, search for relevant memories, and generate a response.
    Always uses a global memory store for all interactions.
    
    Args:
        memory: The Memory object
        message: User's message
        user_id: Ignored - always uses global memory ID
        memory_mode: Ignored - always uses search mode
        output_format: Optional format for structured output
        model: Optional model to use for this specific request
        temperature: Controls randomness (0.0 to 1.0)
        max_tokens: Maximum number of tokens to generate in the response
    
    Returns:
        Dict with response data including:
        - content: The assistant's response
        - memories: Any relevant memories found
        - model: The model used for the response
    """
    # Override any user_id with the global one
    user_id = GLOBAL_MEMORY_ID
    request_id = logging_utils.set_request_id()
    
    # Log the start of the operation
    logger.info(f"Processing chat with memory (request_id={request_id})")
    logger.info(f"Parameters: model={model or OLLAMA_MODEL}, temperature={temperature}, max_tokens={max_tokens}")
    
    # Context for logging
    context = {
        "model": model or OLLAMA_MODEL,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "message_truncated": message[:50] + "..." if len(message) > 50 else message
    }
    
    # Use specified model or fall back to global default
    model_to_use = model or OLLAMA_MODEL
    
    relevant_memories = []
    memories_str = ""
    
    try:
        # Search for relevant memories
        logger.info(f"Searching for relevant memories with message: '{message[:50]}...'")
        start_time = time.time()
        
        search_results = memory.search(query=message, user_id=user_id, limit=20)
        memory_search_duration = (time.time() - start_time) * 1000
        
        relevant_memories = search_results.get("results", [])
        
        # Log memory operation
        logging_utils.log_memory_operation(
            memory_logger,
            operation="search",
            user_id=user_id,
            success=True,
            details={
                "memory_count": len(relevant_memories),
                "query_length": len(message),
                "memory_limit": 20
            },
            duration_ms=memory_search_duration
        )
        
        if relevant_memories:
            logger.info(f"Found {len(relevant_memories)} relevant memories")
            memories_str = "\n".join(f"- {entry['memory']}" for entry in relevant_memories)
        else:
            logger.info("No relevant memories found")
            memories_str = "No relevant memories found."
            
        # Also get some user's recent memories regardless of relevance
        try:
            logger.info("Fetching recent memories as fallback")
            start_time = time.time()
            
            user_memories = memory.get_all(user_id=user_id, limit=5)
            get_all_duration = (time.time() - start_time) * 1000
            
            # Log memory operation
            logging_utils.log_memory_operation(
                memory_logger,
                operation="get_all",
                user_id=user_id,
                success=True,
                details={"memory_count": len(user_memories) if user_memories else 0},
                duration_ms=get_all_duration
            )
            
            if user_memories and not relevant_memories:
                logger.info(f"Using {len(user_memories)} user memories as fallback")
                memories_str = "\n".join(f"- {entry}" for entry in user_memories)
                relevant_memories = [{"memory": memory} for memory in user_memories]
        except Exception as user_mem_error:
            logger.error(f"Error retrieving user memories: {user_mem_error}")
            logging_utils.log_exception(logger, user_mem_error, {"user_id": user_id})
            logging_utils.log_memory_operation(
                memory_logger,
                operation="get_all",
                user_id=user_id,
                success=False,
                error=user_mem_error
            )
    except Exception as e:
        logger.error(f"Error retrieving memories: {e}")
        logging_utils.log_exception(logger, e, {"user_id": user_id, "query": message[:50]})
        logging_utils.log_memory_operation(
            memory_logger,
            operation="search",
            user_id=user_id,
            success=False,
            error=e
        )
        memories_str = "Error retrieving memories, but continuing with chat."
        
    # Generate system prompt with memory context
    system_prompt = f"""You are a helpful AI assistant with memory capabilities.
Answer the question based on the user's query and relevant memories.

User Memories:
{memories_str}

Please be conversational and friendly in your responses.
If referring to a memory, try to naturally incorporate it without explicitly stating 'According to your memory...'""" 
    
    # Create message history for context
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": message}
    ]
    
    try:
        # Import here to avoid circular imports
        from ollama_client import chat_with_ollama
        
        logger.info(f"Sending chat request to Ollama with model: {model_to_use}")
        
        # Send chat request to Ollama with temperature and max_tokens
        result = chat_with_ollama(
            messages=messages,
            model=model_to_use,
            output_format=output_format,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        # Extract assistant response
        if "message" in result and "content" in result["message"]:
            assistant_response = result["message"]["content"]
        else:
            assistant_response = result.get("response", "I couldn't generate a response.")
        
        logger.info(f"Got response from Ollama (length: {len(assistant_response)})")
        
        # Enhanced memory storage - store user and assistant messages separately for better retrieval
        try:
            # Process user message to make it more prominent in vector store
            enhanced_user_message = preprocess_user_message(message)
            
            # Store user message first (with clear prefix for better retrieval)
            metadata = {"active": True, "timestamp": time.time(), "type": "user_message"}
            
            logger.info(f"Storing user message to memory (user_id={user_id})")
            start_time = time.time()
            
            memory.add(
                f"USER INPUT: {enhanced_user_message}",
                user_id=user_id,
                metadata=metadata
            )
            
            user_memory_duration = (time.time() - start_time) * 1000
            
            # Log memory operation
            logging_utils.log_memory_operation(
                memory_logger,
                operation="add_user_message",
                user_id=user_id,
                success=True,
                details={"message_length": len(enhanced_user_message)},
                duration_ms=user_memory_duration
            )
            
            # Update memory counters
            global MEMORY_COUNTER
            MEMORY_COUNTER["active"] += 1
            MEMORY_COUNTER["total"] += 1
            logger.info(f"Successfully stored user message for {user_id}")
            
            # Then store assistant response separately (also with clear prefix)
            metadata = {"active": True, "timestamp": time.time(), "type": "assistant_response"}
            
            logger.info(f"Storing assistant response to memory (user_id={user_id})")
            start_time = time.time()
            
            memory.add(
                f"ASSISTANT RESPONSE: {assistant_response}",
                user_id=user_id,
                metadata=metadata
            )
            
            assistant_memory_duration = (time.time() - start_time) * 1000
            
            # Log memory operation
            logging_utils.log_memory_operation(
                memory_logger,
                operation="add_assistant_response",
                user_id=user_id,
                success=True,
                details={"response_length": len(assistant_response)},
                duration_ms=assistant_memory_duration
            )
            
            # Update memory counters again
            MEMORY_COUNTER["active"] += 1
            MEMORY_COUNTER["total"] += 1
            logger.info(f"Successfully stored assistant response for {user_id}")
            
        except Exception as memory_error:
            logger.error(f"Error adding memory: {memory_error}")
            logging_utils.log_exception(logger, memory_error, {"user_id": user_id})
            logging_utils.log_memory_operation(
                memory_logger,
                operation="add",
                user_id=user_id,
                success=False,
                error=memory_error
            )
            # Continue execution even if memory storage fails
        
        # Return formatted response
        logger.info(f"Returning chat response with {len(relevant_memories)} memories")
        return {
            "content": assistant_response,
            "memories": relevant_memories,
            "model": model_to_use,
            "choices": [
                {"message": {"content": assistant_response}}
            ],
            "conversation_id": user_id
        }
    except Exception as e:
        logger.error(f"Error in chat completion: {e}")
        logging_utils.log_exception(logger, e, context)
        # Re-raise to allow caller to handle
        raise