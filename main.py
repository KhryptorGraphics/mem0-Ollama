import os
import sys
import logging
from api import app
from config import PORT, DEBUG, OLLAMA_HOST
import logging_utils
from ollama_client import OllamaClient

# Configure logging
logger = logging_utils.get_logger(__name__)

def check_ollama_service():
    """Check if Ollama service is running and accessible"""
    client = OllamaClient(host="http://localhost:11434")
    
    if not client.is_server_running():
        logger.error(f"Cannot connect to Ollama at {client.host}")
        logger.error("Please make sure Ollama is installed and running")
        logger.error("Download Ollama from https://ollama.com/download")
        logger.error("Or run python check_model.py for detailed diagnostics")
        return False
    
    logger.info(f"Connected to Ollama at {client.host}")
    return True

def main():
    """Main entry point for the application"""
    logger.info(f"Starting mem0-Ollama server on port {PORT}")
    logger.info(f"Access web interface at http://localhost:{PORT}/")
    
    # Check Ollama service status
    check_ollama_service()
    
    # Start the Flask application
    app.run(host='0.0.0.0', port=PORT, debug=DEBUG)

if __name__ == "__main__":
    main()
