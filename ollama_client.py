import requests
import json
import time
import random
import logging_utils
from config import OLLAMA_HOST
from typing import Dict, List, Any, Optional, Iterator, Union
from functools import wraps

# Get properly configured logger
logger = logging_utils.get_logger(__name__)

# Constants for retry handling
MAX_RETRIES = 3
RETRY_BACKOFF_BASE = 1.5
RETRY_JITTER = 0.1

def retry_with_backoff(max_retries=MAX_RETRIES, backoff_base=RETRY_BACKOFF_BASE, jitter=RETRY_JITTER):
    """Decorator for retrying API calls with exponential backoff and jitter."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            attempt = 0
            while attempt < max_retries:
                try:
                    return func(*args, **kwargs)
                except (requests.ConnectionError, requests.Timeout) as e:
                    attempt += 1
                    if attempt >= max_retries:
                        logger.error(f"All {max_retries} retry attempts failed")
                        raise
                    
                    # Calculate backoff time with jitter
                    backoff = backoff_base ** attempt
                    sleep_time = backoff * (1 + random.uniform(-jitter, jitter))
                    
                    logger.warning(f"Request failed: {str(e)}. Retrying in {sleep_time:.2f}s (attempt {attempt}/{max_retries})")
                    time.sleep(sleep_time)
            return func(*args, **kwargs)  # Final attempt
        return wrapper
    return decorator

class OllamaClient:
    def __init__(self, host=None):
        # Always use the config value if not explicitly overridden
        self.host = host if host is not None else OLLAMA_HOST
        
        # Make sure host has proper URL scheme
        if not self.host.startswith(('http://', 'https://')):
            self.host = f"http://{self.host}"
            
        logger.info(f"Initialized Ollama client with host: {self.host}")
        logger.debug(f"Using configured OLLAMA_HOST from config: {OLLAMA_HOST}")
        
        # Create a session for connection pooling
        self.session = requests.Session()
        
        # Cache for available models
        self._available_models = None
        self._last_models_check = 0
    
    def is_server_running(self) -> bool:
        """Check if the Ollama server is running and accessible."""
        try:
            url = f"{self.host}/api/version"
            response = self.session.get(url, timeout=5)
            if response.status_code == 200:
                return True
            return False
        except Exception:
            return False

    def verify_model_available(self, model_name: str) -> bool:
        """
        Check if a specific model is available.
        
        Args:
            model_name: Name of the model to check, with or without tag
        
        Returns:
            True if available, False otherwise
        """
        available_models = self.get_available_models()
        
        # Handle model names with or without tags
        if model_name in available_models:
            return True
            
        # Check if the model exists with a default tag (:latest)
        if ":" not in model_name and f"{model_name}:latest" in available_models:
            return True
            
        # Check if model name is a prefix of any available model
        for available_model in available_models:
            if available_model.startswith(f"{model_name}:"):
                return True
                
        return False

    @retry_with_backoff()
    def generate(self, model: str, prompt: str, **kwargs) -> str:
        """
        Generate a response from Ollama with improved error handling.
        
        Args:
            model: Name of the model to use
            prompt: Input prompt for generation
            **kwargs: Additional parameters for the API
            
        Returns:
            Generated text or error message
        """
        try:
            # Check if server is running first
            if not self.is_server_running():
                error_msg = "Ollama server is not running or not accessible"
                logger.error(error_msg)
                return f"Error: {error_msg}. Please check if Ollama is running at {self.host}"
                
            # Format request payload according to Ollama API
            url = f"{self.host}/api/generate"
            payload = {
                "model": model,
                "prompt": prompt
            }
            
            # Handle options format according to current Ollama API
            options = {}
            for key, value in kwargs.items():
                if key in ['temperature', 'top_p', 'top_k', 'repeat_penalty']:
                    options[key] = value
                elif key not in ['stream']:
                    # Add other parameters directly to payload
                    payload[key] = value
                    
            if options:
                payload['options'] = options
                
            # Send request with proper headers
            headers = {
                'Content-Type': 'application/json'
            }
            
            logger.info(f"Sending request to Ollama for model: {model}")
            response = self.session.post(url, json=payload, headers=headers, timeout=60)
            response.raise_for_status()
            response_data = response.json()
            
            logger.info(f"Successfully generated response from Ollama")
            return response_data.get('response', '')
        except requests.exceptions.Timeout:
            error_msg = f"Request to Ollama timed out after 60 seconds"
            logger.error(f"{error_msg}")
            return f"Error: {error_msg}. Please try again later."
        except requests.exceptions.ConnectionError:
            error_msg = f"Connection error to Ollama at {self.host}"
            logger.error(f"{error_msg}")
            return f"Error: {error_msg}. Please check if Ollama is running."
        except requests.exceptions.HTTPError as e:
            error_msg = f"HTTP error from Ollama: {e}"
            logger.error(error_msg)
            
            # Try to get detailed error message from response
            try:
                error_detail = e.response.json().get('error', str(e))
                return f"Error from Ollama: {error_detail}"
            except:
                return f"Error: Ollama returned an error: {e}"
        except Exception as e:
            logger.error(f"Error generating response from Ollama: {e}")
            return f"Error: {str(e)}"
    
    @retry_with_backoff()
    def generate_stream(self, model: str, prompt: str, **kwargs) -> Iterator[str]:
        """
        Generate a streaming response from Ollama with improved error handling.
        
        Args:
            model: Name of the model to use
            prompt: Input prompt for generation
            **kwargs: Additional parameters for the API
            
        Yields:
            Generated text chunks or error messages
        """
        try:
            # Check if server is running first
            if not self.is_server_running():
                error_msg = "Ollama server is not running or not accessible"
                logger.error(error_msg)
                yield f"Error: {error_msg}. Please check if Ollama is running at {self.host}"
                return
                
            # Format request payload according to Ollama API
            url = f"{self.host}/api/generate"
            payload = {
                "model": model,
                "prompt": prompt,
                "stream": True
            }
            
            # Handle options format according to current Ollama API
            options = {}
            for key, value in kwargs.items():
                if key in ['temperature', 'top_p', 'top_k', 'repeat_penalty']:
                    options[key] = value
                elif key not in ['stream']:
                    # Add other parameters directly to payload
                    payload[key] = value
                    
            if options:
                payload['options'] = options
                
            # Send request with proper headers
            headers = {
                'Content-Type': 'application/json',
                'Accept': 'application/json'
            }
            
            logger.info(f"Sending streaming request to Ollama for model: {model}")
            response = self.session.post(url, json=payload, headers=headers, stream=True, timeout=60)
            response.raise_for_status()
            
            for line in response.iter_lines():
                if line:
                    try:
                        line_data = json.loads(line.decode('utf-8'))
                        if 'response' in line_data:
                            yield line_data['response']
                        # Also handle potential error messages in stream
                        elif 'error' in line_data:
                            error_msg = line_data['error']
                            logger.error(f"Error in stream: {error_msg}")
                            yield f"Error from Ollama: {error_msg}"
                    except json.JSONDecodeError:
                        logger.warning(f"Failed to decode JSON from Ollama: {line}")
                        continue
            
            logger.info(f"Completed streaming response from Ollama")
        except requests.exceptions.Timeout:
            error_msg = f"Request to Ollama timed out after 60 seconds"
            logger.error(f"{error_msg}")
            yield f"Error: {error_msg}. Please try again later."
        except requests.exceptions.ConnectionError:
            error_msg = f"Connection error to Ollama at {self.host}"
            logger.error(f"{error_msg}")
            yield f"Error: {error_msg}. Please check if Ollama is running."
        except requests.exceptions.HTTPError as e:
            error_msg = f"HTTP error from Ollama: {e}"
            logger.error(error_msg)
            
            # Try to get detailed error message from response
            try:
                error_detail = e.response.json().get('error', str(e))
                yield f"Error from Ollama: {error_detail}"
            except:
                yield f"Error: Ollama returned an error: {e}"
        except Exception as e:
            logger.error(f"Error streaming response from Ollama: {e}")
            yield f"Error: {str(e)}"
    
    @retry_with_backoff()
    def get_available_models(self, force_refresh=False) -> List[str]:
        """
        Get a list of available models from Ollama with caching.
        
        Args:
            force_refresh: If True, bypass cache and fetch fresh data
            
        Returns:
            List of model names
        """
        # Check if we have a cached list that's still valid (less than 60 seconds old)
        current_time = time.time()
        if (not force_refresh and 
            self._available_models is not None and 
            current_time - self._last_models_check < 60):
            return self._available_models
            
        try:
            # Check if server is running first
            if not self.is_server_running():
                logger.error("Cannot fetch models: Ollama server is not running")
                return []
                
            url = f"{self.host}/api/tags"
            logger.info(f"Fetching available models from Ollama")
            
            headers = {'Accept': 'application/json'}
            response = self.session.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            response_data = response.json()
            
            models = response_data.get('models', [])
            model_names = [model.get('name') for model in models]
            
            # Update cache
            self._available_models = model_names
            self._last_models_check = current_time
            
            logger.info(f"Found {len(model_names)} available models")
            return model_names
        except requests.exceptions.Timeout:
            logger.error("Timeout while fetching models from Ollama")
            # Return cached data if available
            return self._available_models if self._available_models else []
        except requests.exceptions.ConnectionError:
            logger.error(f"Connection error when fetching models from Ollama at {self.host}")
            return self._available_models if self._available_models else []
        except Exception as e:
            logger.error(f"Error fetching available models: {e}")
            return self._available_models if self._available_models else []
    
    @retry_with_backoff(max_retries=5)  # More retries for model pulling
    def pull_model(self, model_name: str) -> bool:
        """
        Pull a model from Ollama (download if not available).
        This operation can take a long time for large models.
        
        Args:
            model_name: Name of the model to pull
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Check if server is running first
            if not self.is_server_running():
                logger.error(f"Cannot pull model: Ollama server is not running")
                return False
                
            url = f"{self.host}/api/pull"
            payload = {"name": model_name}
            
            # Set a longer timeout for model pulling
            timeout = 600  # 10 minutes
            
            headers = {
                'Content-Type': 'application/json'
            }
            
            logger.info(f"Pulling model {model_name} from Ollama (this may take a while)")
            response = self.session.post(url, json=payload, headers=headers, timeout=timeout)
            response.raise_for_status()
            
            # Refresh the model list cache after a successful pull
            self._last_models_check = 0  # Force refresh on next get_available_models call
            
            logger.info(f"Successfully pulled model {model_name}")
            return True
        except requests.exceptions.Timeout:
            logger.error(f"Timeout while pulling model {model_name} (operation took too long)")
            return False
        except requests.exceptions.ConnectionError:
            logger.error(f"Connection error when pulling model {model_name}")
            return False
        except requests.exceptions.HTTPError as e:
            # Try to get detailed error message
            try:
                error_detail = e.response.json().get('error', str(e))
                logger.error(f"HTTP error pulling model {model_name}: {error_detail}")
            except:
                logger.error(f"HTTP error pulling model {model_name}: {e}")
            return False
        except Exception as e:
            logger.error(f"Error pulling model {model_name}: {e}")
            return False
