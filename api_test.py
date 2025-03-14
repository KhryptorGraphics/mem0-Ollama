#!/usr/bin/env python3
"""
API Test Script for mem0-ollama

This script provides a direct API test to debug connection issues with Ollama.
It bypasses the main application to isolate if the issue is with Ollama itself
or with the main application's integration.
"""

from flask import Flask, jsonify, request
import requests
import time
import sys
import json
import os
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('api_test')

# Flask app with simple CORS
app = Flask(__name__)

# Default Ollama settings
OLLAMA_HOST = os.environ.get('OLLAMA_HOST', 'http://localhost:11434')
# Force localhost if 0.0.0.0 is being used
if '0.0.0.0' in OLLAMA_HOST:
    OLLAMA_HOST = 'http://localhost:11434'

@app.route('/')
def index():
    """Show a simple status page with information and links to test endpoints"""
    return f"""
    <html>
    <head>
        <title>Ollama API Test</title>
        <style>
            body {{ font-family: sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }}
            h1, h2 {{ color: #0066cc; }}
            pre {{ background: #f0f3f8; padding: 10px; border-radius: 4px; overflow: auto; }}
            .endpoint {{ margin-bottom: 20px; border-bottom: 1px solid #ddd; padding-bottom: 10px; }}
            .status {{ padding: 5px 10px; border-radius: 4px; display: inline-block; }}
            .success {{ background: #d4edda; color: #155724; }}
            .error {{ background: #f8d7da; color: #721c24; }}
        </style>
    </head>
    <body>
        <h1>Ollama API Test</h1>
        <p>This utility helps diagnose API connection issues with Ollama.</p>
        
        <div class="endpoint">
            <h2>Ollama Connection</h2>
            <p>Host: <code>{OLLAMA_HOST}</code></p>
            <p><a href="/test/version">Test Version API</a></p>
            <p><a href="/test/tags">Test Tags API (model list)</a></p>
            <p><a href="/test/generate?model=llama3:latest&prompt=Hello">Test Generate API</a></p>
        </div>
        
        <div class="endpoint">
            <h2>Raw Response</h2>
            <p><a href="/raw/tags">Raw Tags Response</a></p>
        </div>
    </body>
    </html>
    """

@app.route('/test/version')
def test_version():
    """Test the Ollama version endpoint"""
    try:
        url = f"{OLLAMA_HOST}/api/version"
        logger.info(f"Testing Ollama version endpoint at {url}")
        
        start_time = time.time()
        response = requests.get(url, timeout=5)
        elapsed = time.time() - start_time
        
        if response.status_code == 200:
            data = response.json()
            return jsonify({
                "success": True,
                "message": f"Successfully connected to Ollama ({elapsed:.2f}s)",
                "version": data.get('version', 'unknown'),
                "status_code": response.status_code,
                "elapsed_seconds": elapsed
            })
        else:
            return jsonify({
                "success": False,
                "message": f"Received non-200 status code: {response.status_code}",
                "status_code": response.status_code,
                "response_text": response.text,
                "elapsed_seconds": elapsed
            }), 500
    except Exception as e:
        logger.error(f"Error testing version endpoint: {e}")
        return jsonify({
            "success": False,
            "message": f"Error: {str(e)}",
            "error_type": str(type(e).__name__)
        }), 500

@app.route('/test/tags')
def test_tags():
    """Test the Ollama tags (model list) endpoint"""
    try:
        url = f"{OLLAMA_HOST}/api/tags"
        logger.info(f"Testing Ollama tags endpoint at {url}")
        
        headers = {
            'Accept': 'application/json',
            'User-Agent': 'APITest/1.0'
        }
        
        start_time = time.time()
        response = requests.get(url, headers=headers, timeout=10)
        elapsed = time.time() - start_time
        
        if response.status_code == 200:
            try:
                data = response.json()
                models = data.get('models', [])
                model_names = [model.get('name') for model in models]
                
                return jsonify({
                    "success": True,
                    "message": f"Successfully listed {len(model_names)} models ({elapsed:.2f}s)",
                    "models": model_names,
                    "status_code": response.status_code,
                    "elapsed_seconds": elapsed
                })
            except json.JSONDecodeError as je:
                return jsonify({
                    "success": False,
                    "message": f"Got 200 status but couldn't parse JSON: {je}",
                    "response_text": response.text[:500],
                    "elapsed_seconds": elapsed
                }), 500
        else:
            return jsonify({
                "success": False,
                "message": f"Received non-200 status code: {response.status_code}",
                "status_code": response.status_code,
                "response_text": response.text,
                "elapsed_seconds": elapsed
            }), 500
    except Exception as e:
        logger.error(f"Error testing tags endpoint: {e}")
        return jsonify({
            "success": False,
            "message": f"Error: {str(e)}",
            "error_type": str(type(e).__name__)
        }), 500

@app.route('/test/generate')
def test_generate():
    """Test the Ollama generate endpoint"""
    model = request.args.get('model', 'llama3:latest')
    prompt = request.args.get('prompt', 'Hello, how are you?')
    
    try:
        url = f"{OLLAMA_HOST}/api/generate"
        logger.info(f"Testing Ollama generate endpoint with model {model}")
        
        headers = {
            'Content-Type': 'application/json',
            'Accept': 'application/json',
            'User-Agent': 'APITest/1.0'
        }
        
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False
        }
        
        start_time = time.time()
        response = requests.post(url, headers=headers, json=payload, timeout=15)
        elapsed = time.time() - start_time
        
        if response.status_code == 200:
            try:
                data = response.json()
                return jsonify({
                    "success": True,
                    "message": f"Successfully generated response ({elapsed:.2f}s)",
                    "response": data.get('response', ''),
                    "status_code": response.status_code,
                    "elapsed_seconds": elapsed
                })
            except json.JSONDecodeError:
                return jsonify({
                    "success": False,
                    "message": "Got 200 status but couldn't parse JSON",
                    "response_text": response.text[:500],
                    "elapsed_seconds": elapsed
                }), 500
        else:
            return jsonify({
                "success": False,
                "message": f"Received non-200 status code: {response.status_code}",
                "status_code": response.status_code,
                "response_text": response.text,
                "elapsed_seconds": elapsed
            }), 500
    except Exception as e:
        logger.error(f"Error testing generate endpoint: {e}")
        return jsonify({
            "success": False,
            "message": f"Error: {str(e)}",
            "error_type": str(type(e).__name__)
        }), 500

@app.route('/raw/tags')
def raw_tags():
    """Get the raw response from the tags endpoint"""
    try:
        url = f"{OLLAMA_HOST}/api/tags"
        logger.info(f"Fetching raw tags response from {url}")
        
        response = requests.get(url, timeout=10)
        
        # Return raw response details
        return jsonify({
            "status_code": response.status_code,
            "headers": dict(response.headers),
            "content_type": response.headers.get('Content-Type'),
            "text": response.text,
            "size": len(response.content),
            "url": url
        })
    except Exception as e:
        logger.error(f"Error fetching raw tags: {e}")
        return jsonify({
            "error": str(e),
            "type": str(type(e).__name__)
        }), 500

if __name__ == '__main__':
    # Get port from command line or use default 5001
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 5001
    
    print(f"Starting API test server on http://localhost:{port}")
    print(f"Using Ollama at: {OLLAMA_HOST}")
    print("Press Ctrl+C to exit")
    
    app.run(host='0.0.0.0', port=port, debug=True)
