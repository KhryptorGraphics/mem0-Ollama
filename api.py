# Importing necessary libraries
from flask import Flask, request, jsonify, send_from_directory, Response, stream_with_context
import logging
import os
import json
import time
import re
from datetime import datetime
from config import *
from memory_utils import MemoryManager
from ollama_client import OllamaClient
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

# Initialize memory manager and Ollama client
memory_manager = MemoryManager()
ollama_client = OllamaClient()

# Pattern to remove thinking tags
thinking_pattern = re.compile(r'<think>.*?</think>', re.DOTALL)

def remove_thinking_tags(text):
    """Remove <think>...</think> tags from the response"""
    return thinking_pattern.sub('', text).strip()

@app.route('/')
def index():
    return send_from_directory('.', 'model_test.html')

@app.route('/direct')
def direct_index():
    return send_from_directory('.', 'direct_test.html')

@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.json
    
    if not data or 'message' not in data:
        return jsonify({'error': 'No message provided'}), 400
    
    message = data['message']
    system_prompt = data.get('system_prompt', DEFAULT_SYSTEM_PROMPT)
    model = data.get('model', DEFAULT_MODEL)
    memory_id = data.get('memory_id', 'default')
    stream = data.get('stream', True)
    temperature = data.get('temperature', 0.7)
    top_p = data.get('top_p', 0.9)
    max_tokens = data.get('max_tokens', 2000)
    n_predict = data.get('n_predict', max_tokens)
    stop_sequences = data.get('stop', [])
    
    # Track active memories
    memory_manager.reset_active_memories(memory_id)
    
    # Retrieve relevant memories for the current message
    memories = memory_manager.get_relevant_memories(
        message, 
        memory_id=memory_id, 
        limit=MAX_MEMORIES
    )
    
    # Construct the context with memories and conversation history
    # Mark retrieved memories as active
    memory_context = ""
    if memories:
        memory_context = "Relevant information from your memory:\n\n"
        for memory in memories:
            memory_manager.mark_memory_active(memory['id'], memory_id)
            memory_context += f"[Memory from {memory['metadata'].get('timestamp', 'unknown time')}]\n{memory['text']}\n\n"
    
    context = f"{system_prompt}\n\n{memory_context}\n\nUser: {message}\n\nAssistant:"
    
    if stream:
        def generate():
            full_response = ""
            for chunk in ollama_client.generate_stream(
                model=model,
                prompt=context,
                temperature=temperature,
                top_p=top_p,
                n_predict=n_predict,
                stop=stop_sequences
            ):
                # Remove thinking tags if present
                clean_chunk = remove_thinking_tags(chunk)
                if clean_chunk:
                    full_response += clean_chunk
                    yield f"data: {json.dumps({'text': clean_chunk})}\n\n"
            
            # Store the memory after completion
            store_memory(message, full_response, memory_id)
            
            yield f"data: [DONE]\n\n"
        
        return Response(stream_with_context(generate()), mimetype='text/event-stream')
    else:
        response_text = ollama_client.generate(
            model=model,
            prompt=context,
            temperature=temperature,
            top_p=top_p,
            n_predict=n_predict,
            stop=stop_sequences
        )
        
        # Clean the response text of any thinking tags
        clean_response = remove_thinking_tags(response_text)
        
        # Store the memory
        store_memory(message, clean_response, memory_id)
        
        return jsonify({
            'text': clean_response,
            'model': model
        })

@app.route('/api/memories', methods=['GET'])
def get_memories():
    memory_id = request.args.get('memory_id', 'default')
    limit = int(request.args.get('limit', 100))
    offset = int(request.args.get('offset', 0))
    
    memories = memory_manager.list_memories(memory_id, limit, offset)
    
    return jsonify({
        'memories': memories,
        'total': len(memories),
        'active': memory_manager.get_active_memories(memory_id)
    })

@app.route('/api/memory', methods=['GET'])
def get_memory():
    memory_id = request.args.get('memory_id', 'default')
    memory_item_id = request.args.get('id')
    
    if not memory_item_id:
        return jsonify({'error': 'No memory ID provided'}), 400
    
    memory = memory_manager.get_memory(memory_item_id, memory_id)
    
    if not memory:
        return jsonify({'error': 'Memory not found'}), 404
    
    return jsonify(memory)

@app.route('/api/memory/active', methods=['GET'])
def get_active_memories():
    memory_id = request.args.get('memory_id', 'default')
    
    active_memories = memory_manager.get_active_memories(memory_id)
    
    return jsonify({
        'active_memories': active_memories,
        'count': len(active_memories)
    })

def store_memory(user_message, assistant_response, memory_id='default'):
    timestamp = datetime.now().isoformat()
    
    # Store user message
    memory_manager.add_memory(
        f"User said: {user_message}",
        memory_id=memory_id,
        metadata={
            'timestamp': timestamp,
            'type': 'user_message'
        }
    )
    
    # Store assistant response
    memory_manager.add_memory(
        f"I responded: {assistant_response}",
        memory_id=memory_id,
        metadata={
            'timestamp': timestamp,
            'type': 'assistant_response'
        }
    )
    
    logger.info(f"Stored memory: User and Assistant conversation in {memory_id}")

@app.route('/api/tags', methods=['GET'])
def ollama_tags_proxy():
    """Proxy Ollama tags API to support Docker container access."""
    try:
        logger.info("Proxying request to Ollama /api/tags endpoint")
        import requests
        response = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=10)

        if response.status_code == 200:
            data = response.json()
            logger.info(f"Ollama tags proxy success: {len(data.get('models', []))} models found")
            return jsonify(data)
        else:
            logger.error(f"Ollama tags proxy error: {response.status_code}")
            return jsonify({"error": "Failed to reach Ollama"}), 500
    except Exception as e:
        logger.error(f"Ollama tags proxy exception: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/pull', methods=['POST'])
def ollama_pull_proxy():
    """Proxy Ollama pull API to support Docker container access."""
    try:
        logger.info("Proxying request to Ollama /api/pull endpoint")
        import requests
        data = request.json
        response = requests.post(f"{OLLAMA_HOST}/api/pull", json=data, timeout=300)

        if response.status_code == 200:
            logger.info(f"Ollama pull proxy success")
            return jsonify(response.json())
        else:
            logger.error(f"Ollama pull proxy error: {response.status_code}")
            return jsonify({"error": "Failed to reach Ollama"}), 500
    except Exception as e:
        logger.error(f"Ollama pull proxy exception: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=PORT, debug=True)
