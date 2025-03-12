import requests
import json
import logging
from config import OLLAMA_HOST

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

class OllamaClient:
    def __init__(self, host=None):
        self.host = host or OLLAMA_HOST
        logger.info(f"Initialized Ollama client with host: {self.host}")
    
    def generate(self, model, prompt, **kwargs):
        """Generate a response from Ollama"""
        try:
            url = f"{self.host}/api/generate"
            payload = {
                "model": model,
                "prompt": prompt,
                **kwargs
            }
            
            logger.info(f"Sending request to Ollama for model: {model}")
            response = requests.post(url, json=payload)
            response.raise_for_status()
            response_data = response.json()
            
            logger.info(f"Successfully generated response from Ollama")
            return response_data.get('response', '')
        except Exception as e:
            logger.error(f"Error generating response from Ollama: {e}")
            return f"Error: {str(e)}"
    
    def generate_stream(self, model, prompt, **kwargs):
        """Generate a streaming response from Ollama"""
        try:
            url = f"{self.host}/api/generate"
            payload = {
                "model": model,
                "prompt": prompt,
                "stream": True,
                **kwargs
            }
            
            logger.info(f"Sending streaming request to Ollama for model: {model}")
            response = requests.post(url, json=payload, stream=True)
            response.raise_for_status()
            
            for line in response.iter_lines():
                if line:
                    try:
                        line_data = json.loads(line.decode('utf-8'))
                        if 'response' in line_data:
                            yield line_data['response']
                    except json.JSONDecodeError:
                        logger.warning(f"Failed to decode JSON from Ollama: {line}")
                        continue
            
            logger.info(f"Completed streaming response from Ollama")
        except Exception as e:
            logger.error(f"Error streaming response from Ollama: {e}")
            yield f"Error: {str(e)}"
    
    def get_available_models(self):
        """Get a list of available models from Ollama"""
        try:
            url = f"{self.host}/api/tags"
            logger.info(f"Fetching available models from Ollama")
            response = requests.get(url)
            response.raise_for_status()
            response_data = response.json()
            
            models = response_data.get('models', [])
            model_names = [model.get('name') for model in models]
            logger.info(f"Found {len(model_names)} available models")
            return model_names
        except Exception as e:
            logger.error(f"Error fetching available models: {e}")
            return []
    
    def pull_model(self, model_name):
        """Pull a model from Ollama"""
        try:
            url = f"{self.host}/api/pull"
            payload = {"name": model_name}
            
            logger.info(f"Pulling model {model_name} from Ollama")
            response = requests.post(url, json=payload)
            response.raise_for_status()
            
            logger.info(f"Successfully pulled model {model_name}")
            return True
        except Exception as e:
            logger.error(f"Error pulling model {model_name}: {e}")
            return False
