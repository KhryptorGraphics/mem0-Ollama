# Simple direct Ollama server
from flask import Flask, request, jsonify, send_from_directory, Response, stream_with_context
import requests
import logging
import os
import json
import time
from datetime import datetime
from config import *
from flask_cors import CORS

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

@app.route('/')
def index():
    return send_from_directory('.', 'direct_test.html')

@app.route('/api/generate', methods=['POST'])
def generate():
    data = request.json
    
    if not data or 'prompt' not in data or 'model' not in data:
        return jsonify({'error': 'Missing required parameters'}), 400
    
    stream = data.get('stream', False)
    
    # Forward the request directly to Ollama
    try:
        ollama_url = f"{OLLAMA_HOST}/api/generate"
        
        if stream:
            def generate_stream():
                response = requests.post(ollama_url, json=data, stream=True)
                response.raise_for_status()
                
                for line in response.iter_lines():
                    if line:
                        yield line + b'\n'
                        
            return Response(generate_stream(), mimetype='application/json')
        else:
            response = requests.post(ollama_url, json=data)
            response.raise_for_status()
            return jsonify(response.json())
    except Exception as e:
        logger.error(f"Error forwarding request to Ollama: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/tags', methods=['GET'])
def ollama_tags():
    """Get available Ollama models"""
    try:
        ollama_url = f"{OLLAMA_HOST}/api/tags"
        response = requests.get(ollama_url)
        response.raise_for_status()
        return jsonify(response.json())
    except Exception as e:
        logger.error(f"Error getting Ollama tags: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    logger.info(f"Starting direct Ollama server on port {PORT}")
    logger.info(f"Using Ollama at {OLLAMA_HOST}")
    
    app.run(host='0.0.0.0', port=PORT, debug=True)
