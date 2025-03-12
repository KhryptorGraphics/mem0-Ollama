"""
Ollama API client functions for mem0 + Ollama integration

This module provides functions for interacting with the Ollama API,
with enhanced logging and error handling.
"""

import time
import requests
import logging
from typing import List, Dict, Any, Optional, Union, Tuple

from config import OLLAMA_HOST, OLLAMA_MODEL
import logging_utils

# Set up specialized logger for API calls
logger = logging_utils.get_logger(__name__)
api_logger = logging_utils.get_logger("api.ollama")

# Default retry settings
DEFAULT_TIMEOUT = 20  # seconds
MAX_RETRIES = 3
RETRY_DELAY = 1.0  # seconds

@logging_utils.timed_operation(operation="get_available_models")
def get_available_models(timeout: int = DEFAULT_TIMEOUT, retries: int = MAX_RETRIES) -> List[Dict[str, Any]]:
    """
    Get available models from Ollama API with retry logic.
    
    Args:
        timeout: Request timeout in seconds
        retries: Maximum number of retry attempts
        
    Returns:
        List of model details
    """
    endpoint = "/api/tags"
    url = f"{OLLAMA_HOST}{endpoint}"
    
    attempt = 0
    while attempt <= retries:
        attempt += 1
        request_id = logging_utils.set_request_id()
        
        try:
            start_time = time.time()
            
            # Log the API call
            logging_utils.log_api_call(
                api_logger,
                method="GET",
                url=url,
                headers=None
            )
            
            # Make the request
            response = requests.get(url, timeout=timeout)
            duration_ms = (time.time() - start_time) * 1000
            
            # Process response
            if response.status_code == 200:
                data = response.json()
                
                # Log successful response
                logging_utils.log_api_call(
                    api_logger,
                    method="GET",
                    url=url,
                    response=data,
                    status_code=response.status_code,
                    duration_ms=duration_ms
                )
                
                # Process models
                models = data.get("models", [])
                logger.info(f"Retrieved {len(models)} models from Ollama")
                
                model_details = []
                for model in models:
                    name = model.get("name")
                    # Extract details or provide defaults
                    parameter_size = model.get("details", {}).get("parameter_size", "unknown")
                    quantization = model.get("details", {}).get("quantization_level", "unknown") 
                    families = model.get("details", {}).get("families", [])
                    
                    # Create model info object
                    details = {
                        "id": name,
                        "name": name,
                        "size": model.get("size", 0),
                        "parameter_size": parameter_size,
                        "quantization": quantization,
                        "families": families,
                        # Add raw model data for better compatibility
                        "raw_model": model
                    }
                    model_details.append(details)
                    logger.debug(f"Added model: {name} ({parameter_size})")
                
                return model_details
            else:
                # Log error response
                error_msg = f"Failed to get models: Status {response.status_code}"
                logging_utils.log_api_call(
                    api_logger,
                    method="GET",
                    url=url,
                    response=response.text,
                    status_code=response.status_code,
                    error=Exception(error_msg),
                    duration_ms=duration_ms
                )
                
                # Retry only on certain status codes
                if response.status_code in (408, 429, 500, 502, 503, 504) and attempt <= retries:
                    backoff = RETRY_DELAY * (2 ** (attempt - 1))  # exponential backoff
                    logger.warning(f"Retrying after {backoff:.2f}s (attempt {attempt}/{retries})")
                    time.sleep(backoff)
                    continue
                else:
                    logger.error(f"Failed to get models: Status {response.status_code}, Response: {response.text}")
                    return []
                    
        except requests.Timeout as e:
            # Log timeout error
            error_msg = f"Request timed out after {timeout}s"
            logging_utils.log_api_call(
                api_logger,
                method="GET",
                url=url,
                error=e,
                status_code=408
            )
            
            if attempt <= retries:
                backoff = RETRY_DELAY * (2 ** (attempt - 1))
                logger.warning(f"Request timed out. Retrying after {backoff:.2f}s (attempt {attempt}/{retries})")
                time.sleep(backoff)
                continue
            else:
                logger.error(f"Request timed out after {retries} attempts: {str(e)}")
                logging_utils.log_exception(logger, e, {"url": url, "timeout": timeout})
                return []
                
        except requests.RequestException as e:
            # Log request error
            logging_utils.log_api_call(
                api_logger,
                method="GET",
                url=url,
                error=e,
                status_code=500  # Assuming 500 for network errors
            )
            
            if attempt <= retries:
                backoff = RETRY_DELAY * (2 ** (attempt - 1))
                logger.warning(f"Request error. Retrying after {backoff:.2f}s (attempt {attempt}/{retries})")
                time.sleep(backoff)
                continue
            else:
                logger.error(f"Request error after {retries} attempts: {str(e)}")
                logging_utils.log_exception(logger, e, {"url": url})
                return []
                
        except Exception as e:
            # Log unexpected error
            logging_utils.log_api_call(
                api_logger,
                method="GET",
                url=url,
                error=e,
                status_code=500
            )
            
            logging_utils.log_exception(logger, e, {"url": url})
            logger.error(f"Unexpected error while fetching models: {e}")
            return []
            
    # If we get here, all retries failed
    logger.error(f"All {retries} attempts to get models failed")
    return []

@logging_utils.timed_operation(operation="check_ollama")
def check_ollama(timeout: int = DEFAULT_TIMEOUT) -> Tuple[bool, str]:
    """
    Check if Ollama is running and has the required model.
    
    Args:
        timeout: Request timeout in seconds
        
    Returns:
        Tuple of (success, message)
    """
    endpoint = "/api/tags"
    url = f"{OLLAMA_HOST}{endpoint}"
    request_id = logging_utils.set_request_id()
    
    try:
        start_time = time.time()
        
        # Log the API call
        logging_utils.log_api_call(
            api_logger,
            method="GET",
            url=url
        )
        
        # Make the request
        response = requests.get(url, timeout=timeout)
        duration_ms = (time.time() - start_time) * 1000
        
        # Log response
        logging_utils.log_api_call(
            api_logger,
            method="GET",
            url=url,
            response=response.json() if response.status_code == 200 else response.text,
            status_code=response.status_code,
            duration_ms=duration_ms
        )
        
        if response.status_code != 200:
            error_msg = f"Ollama is not running or not responding correctly at {OLLAMA_HOST}"
            logger.error(error_msg)
            return False, error_msg
        
        # Check if the model is available
        models = response.json().get("models", [])
        model_names = [model.get("name") for model in models]
        
        if not models:
            msg = f"No models found in Ollama at {OLLAMA_HOST}"
            logger.warning(msg)
            return False, msg
            
        if OLLAMA_MODEL not in model_names and f"{OLLAMA_MODEL}:latest" not in model_names:
            msg = f"Model {OLLAMA_MODEL} not found in Ollama. Available models: {', '.join(model_names[:5])}"
            if len(model_names) > 5:
                msg += f" (and {len(model_names) - 5} more)"
            logger.warning(msg)
            logger.info(f"You may need to pull the model using: ollama pull {OLLAMA_MODEL}")
            return False, msg
        
        msg = f"Ollama is running at {OLLAMA_HOST} with model {OLLAMA_MODEL} available"
        logger.info(msg)
        return True, msg
        
    except requests.Timeout as e:
        error_msg = f"Connection to Ollama timed out after {timeout}s"
        logging_utils.log_api_call(
            api_logger,
            method="GET",
            url=url,
            error=e,
            status_code=408
        )
        logger.error(error_msg)
        logging_utils.log_exception(logger, e, {"url": url, "timeout": timeout})
        return False, error_msg
        
    except requests.RequestException as e:
        error_msg = f"Error connecting to Ollama: {str(e)}"
        logging_utils.log_api_call(
            api_logger,
            method="GET",
            url=url,
            error=e,
            status_code=500
        )
        logger.error(error_msg)
        logging_utils.log_exception(logger, e, {"url": url})
        return False, error_msg
        
    except Exception as e:
        error_msg = f"Unexpected error checking Ollama: {str(e)}"
        logging_utils.log_exception(logger, e, {"url": url})
        logger.error(error_msg)
        return False, error_msg

@logging_utils.timed_operation(operation="chat_with_ollama")
def chat_with_ollama(
    messages: List[Dict[str, str]], 
    model: str = OLLAMA_MODEL,
    output_format: Optional[Union[str, Dict]] = None,
    temperature: float = 0.7,
    max_tokens: int = 2000,
    timeout: int = 60,  # Longer timeout for chat completions
    retries: int = MAX_RETRIES
) -> Dict[str, Any]:
    """
    Send a chat request to Ollama's API with retry logic.
    
    Args:
        messages: List of message objects (role, content)
        model: Model to use for chat
        output_format: Optional format for structured output
        temperature: Controls randomness (0.0 to 1.0)
        max_tokens: Maximum number of tokens to generate in the response
        timeout: Request timeout in seconds
        retries: Maximum number of retry attempts
    
    Returns:
        Dict with the Ollama API response
    """
    # Prepare request payload
    request_payload = {
        "model": model,
        "stream": False,
        "options": {
            "temperature": float(temperature),  # Ensure it's a float
            "num_predict": int(max_tokens)      # Ollama's parameter for max tokens
        },
        "messages": messages
    }
    
    # Add format for structured output if specified
    if output_format:
        request_payload["format"] = output_format
        format_desc = output_format if isinstance(output_format, str) else "custom JSON schema"
        logger.info(f"Using structured output format: {format_desc}")
    
    endpoint = "/api/chat"
    url = f"{OLLAMA_HOST}{endpoint}"
    
    # Context for logging
    user_message = next((m["content"] for m in messages if m["role"] == "user"), None)
    context = {
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "message_count": len(messages)
    }
    if user_message:
        # Truncate long messages for logging
        context["user_message"] = (user_message[:50] + "...") if len(user_message) > 50 else user_message
    
    attempt = 0
    while attempt <= retries:
        attempt += 1
        request_id = logging_utils.set_request_id()
        
        try:
            start_time = time.time()
            
            # Log the API call
            logging_utils.log_api_call(
                api_logger,
                method="POST",
                url=url,
                payload=request_payload
            )
            
            # Make the request
            response = requests.post(url, json=request_payload, timeout=timeout)
            duration_ms = (time.time() - start_time) * 1000
            
            # Process response
            if response.status_code == 200:
                try:
                    result = response.json()
                    
                    # Log successful response (with truncated content)
                    log_response = result.copy() if isinstance(result, dict) else result
                    if isinstance(log_response, dict) and "message" in log_response and "content" in log_response["message"]:
                        content = log_response["message"]["content"]
                        if len(content) > 200:
                            log_response["message"]["content"] = content[:200] + "... [truncated]"
                            
                    logging_utils.log_api_call(
                        api_logger,
                        method="POST",
                        url=url,
                        response=log_response,
                        status_code=response.status_code,
                        duration_ms=duration_ms
                    )
                    
                    logger.info(f"Ollama chat completion successful in {duration_ms:.2f}ms")
                    return result
                    
                except ValueError as e:
                    # JSON parse error
                    error_msg = f"Invalid JSON in Ollama response: {str(e)}"
                    logging_utils.log_api_call(
                        api_logger,
                        method="POST",
                        url=url,
                        response=response.text[:500],
                        status_code=response.status_code,
                        error=e,
                        duration_ms=duration_ms
                    )
                    
                    logger.error(error_msg)
                    logging_utils.log_exception(logger, e, context)
                    raise ValueError(error_msg)
            else:
                # Log error response
                error_msg = f"Ollama API error: Status {response.status_code}"
                logging_utils.log_api_call(
                    api_logger,
                    method="POST",
                    url=url,
                    response=response.text,
                    status_code=response.status_code,
                    error=Exception(error_msg),
                    duration_ms=duration_ms
                )
                
                # Retry only on certain status codes
                if response.status_code in (408, 429, 500, 502, 503, 504) and attempt <= retries:
                    backoff = RETRY_DELAY * (2 ** (attempt - 1))  # exponential backoff
                    logger.warning(f"Retrying after {backoff:.2f}s (attempt {attempt}/{retries})")
                    time.sleep(backoff)
                    continue
                else:
                    error_msg = f"Ollama API error: Status {response.status_code}, Response: {response.text[:200]}"
                    logger.error(error_msg)
                    raise requests.HTTPError(error_msg)
                
        except requests.Timeout as e:
            # Log timeout error
            error_msg = f"Ollama request timed out after {timeout}s"
            logging_utils.log_api_call(
                api_logger,
                method="POST",
                url=url,
                payload=request_payload,
                error=e,
                status_code=408
            )
            
            if attempt <= retries:
                backoff = RETRY_DELAY * (2 ** (attempt - 1))
                logger.warning(f"Request timed out. Retrying after {backoff:.2f}s (attempt {attempt}/{retries})")
                time.sleep(backoff)
                continue
            else:
                logger.error(f"Ollama chat request timed out after {retries} attempts")
                logging_utils.log_exception(logger, e, context)
                raise TimeoutError(f"Ollama chat request timed out after {retries} attempts") from e
                
        except requests.RequestException as e:
            # Log request error
            logging_utils.log_api_call(
                api_logger,
                method="POST",
                url=url,
                payload=request_payload,
                error=e,
                status_code=500
            )
            
            if attempt <= retries:
                backoff = RETRY_DELAY * (2 ** (attempt - 1))
                logger.warning(f"Request error. Retrying after {backoff:.2f}s (attempt {attempt}/{retries})")
                time.sleep(backoff)
                continue
            else:
                logger.error(f"Ollama chat request failed after {retries} attempts: {str(e)}")
                logging_utils.log_exception(logger, e, context)
                raise
                
        except Exception as e:
            # Log unexpected error
            logging_utils.log_api_call(
                api_logger,
                method="POST",
                url=url,
                payload=request_payload,
                error=e,
                status_code=500
            )
            
            logging_utils.log_exception(logger, e, context)
            logger.error(f"Unexpected error in chat request: {str(e)}")
            raise
            
    # If we get here, all retries failed
    error_msg = f"All {retries} attempts to complete chat request failed"
    logger.error(error_msg)
    raise RuntimeError(error_msg)