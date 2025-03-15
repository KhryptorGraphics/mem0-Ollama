# Importing necessary libraries
from flask import Flask, request, jsonify, send_from_directory, Response, stream_with_context
import os
import json
import time
import re
from datetime import datetime
from config import *
# Use the improved mem0ai integration with Qdrant support
from final_mem0_adapter import Mem0Adapter as MemoryManager
from ollama_client import OllamaClient
from flask_cors import CORS
import logging_utils
# Import health check utilities
from health_check import get_system_health, check_ollama_health, check_mem0_health, check_qdrant_health
# Import the Qdrant MCP conversation store
from qdrant_conversation_store import conversation_store

app = Flask(__name__)
# More secure CORS configuration with specific allowed origins
allowed_origins = os.environ.get('ALLOWED_ORIGINS', 'http://localhost:8000,http://127.0.0.1:8000').split(',')
CORS(app, resources={r"/*": {"origins": allowed_origins}})

# Get properly configured logger
logger = logging_utils.get_logger(__name__)

# Initialize memory manager and Ollama client
memory_manager = MemoryManager(ollama_host=OLLAMA_HOST, mem0_host=MEM0_HOST, default_model=DEFAULT_MODEL)
# Make sure the Ollama host is set to localhost:11434
if "localhost" not in OLLAMA_HOST and "127.0.0.1" not in OLLAMA_HOST:
    logger.warning(f"Ollama host is set to {OLLAMA_HOST}, which may not be correct. Using http://localhost:11434 instead.")
    ollama_host = "http://localhost:11434"
else:
    ollama_host = OLLAMA_HOST

# Initialize client with explicit host
ollama_client = OllamaClient(host=ollama_host)
logger.info(f"API initialized with Ollama host: {ollama_host}")

# Pattern to extract thinking tags - supports both <think> and <thinking> formats
thinking_pattern = re.compile(r'<think(?:ing)?>(.+?)</think(?:ing)?>', re.DOTALL)

def remove_thinking_tags(text):
    """
    Remove <think>...</think> or <thinking>...</thinking> tags from the response.
    Also returns the extracted thinking content if it exists.
    
    Returns:
        Cleaned text with thinking tags removed
    """
    thinking_content = []
    
    # Extract all thinking tags content before removing
    for match in thinking_pattern.finditer(text):
        if match.group(1):
            thinking_content.append(match.group(1).strip())
    
    # Remove thinking tags
    cleaned_text = thinking_pattern.sub('', text).strip()
    
    # Return the cleaned text
    return cleaned_text
    
def extract_thinking_content(text):
    """
    Extract thinking content from text with thinking tags.
    
    Args:
        text: Original text that may contain thinking tags
        
    Returns:
        List of extracted thinking content
    """
    thinking_content = []
    
    # Extract all thinking tags content
    for match in thinking_pattern.finditer(text):
        if match.group(1):
            thinking_content.append(match.group(1).strip())
            
    return thinking_content

@app.route('/')
def index():
    # Serve the new chat interface as the main page
    return send_from_directory('.', 'chat_interface.html')

@app.route('/model_test')
def model_test():
    # Keep the original model_test page available at a different route
    return send_from_directory('.', 'model_test.html')

@app.route('/direct')
def direct_index():
    return send_from_directory('.', 'direct_test.html')

@app.route('/debug')
def debug_ollama():
    """Debug page for Ollama connection"""
    return send_from_directory('.', 'debug_ollama.html')

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
    if hasattr(memory_manager, 'reset_active_memories'):
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
            if hasattr(memory_manager, 'mark_memory_active'):
                memory_manager.mark_memory_active(memory['id'], memory_id)
            memory_context += f"[Memory from {memory.get('metadata', {}).get('timestamp', 'unknown time')}]\n{memory.get('text', '')}\n\n"
    
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
    source = request.args.get('source')  # Optional source filter
    
    # Group memories by source if no specific source is requested
    if source:
        # Get memories with specific source filter
        memories = memory_manager.list_memories(memory_id, limit, offset, source=source)
        
        # Handle if memory_manager.get_active_memories is not implemented
        active_memories = []
        if hasattr(memory_manager, 'get_active_memories'):
            active_memories = memory_manager.get_active_memories(memory_id)
        
        return jsonify({
            'memories': memories,
            'total': len(memories),
            'active': active_memories,
            'source': source
        })
    else:
        # Group memories by source
        mem0ai_memories = memory_manager.list_memories(memory_id, limit, offset, source='mem0ai')
        fallback_memories = memory_manager.list_memories(memory_id, limit, offset, source='fallback')
        unknown_memories = memory_manager.list_memories(memory_id, limit, offset, source='unknown')
        
        # Handle if memory_manager.get_active_memories is not implemented
        active_memories = []
        if hasattr(memory_manager, 'get_active_memories'):
            active_memories = memory_manager.get_active_memories(memory_id)
        
        # All memories combined (for backward compatibility)
        all_memories = mem0ai_memories + fallback_memories + unknown_memories
        
        return jsonify({
            'memories': all_memories,  # For backward compatibility
            'total': len(all_memories),
            'active': active_memories,
            'grouped_memories': {
                'mem0ai': mem0ai_memories,
                'fallback': fallback_memories,
                'unknown': unknown_memories
            },
            'counts': {
                'mem0ai': len(mem0ai_memories),
                'fallback': len(fallback_memories),
                'unknown': len(unknown_memories)
            }
        })

@app.route('/api/memory', methods=['GET'])
def get_memory():
    memory_id = request.args.get('memory_id', 'default')
    memory_item_id = request.args.get('id')
    
    if not memory_item_id:
        return jsonify({'error': 'No memory ID provided'}), 400
    
    # Handle if memory_manager.get_memory is not implemented
    if hasattr(memory_manager, 'get_memory'):
        memory = memory_manager.get_memory(memory_item_id, memory_id)
        
        if not memory:
            return jsonify({'error': 'Memory not found'}), 404
        
        return jsonify(memory)
    else:
        return jsonify({'error': 'Memory retrieval not implemented'}), 501

@app.route('/api/memory/active', methods=['GET'])
def get_active_memories():
    memory_id = request.args.get('memory_id', 'default')
    
    # Handle if memory_manager.get_active_memories is not implemented
    if hasattr(memory_manager, 'get_active_memories'):
        active_memories = memory_manager.get_active_memories(memory_id)
        
        return jsonify({
            'active_memories': active_memories,
            'count': len(active_memories)
        })
    else:
        return jsonify({
            'active_memories': [],
            'count': 0
        })

@app.route('/api/conversations/search', methods=['GET'])
def search_conversations():
    """Search for conversations stored in Qdrant MCP."""
    try:
        if not QDRANT_MCP_ENABLED:
            return jsonify({
                'error': 'Qdrant MCP conversation storage is not enabled'
            }), 400
            
        query = request.args.get('query', '')
        limit = int(request.args.get('limit', 5))
        
        if not query:
            return jsonify({'error': 'No search query provided'}), 400
            
        # Use conversation store to search
        success, results, error = conversation_store.find_conversations(
            query=query,
            limit=limit
        )
        
        if success:
            return jsonify({
                'results': results or [],
                'count': len(results) if results else 0,
                'query': query
            })
        else:
            return jsonify({
                'error': error or 'Unknown error searching conversations',
                'query': query
            }), 500
            
    except Exception as e:
        logger.error(f"Error searching conversations: {e}")
        return create_error_response(
            message="Error searching conversations", 
            status_code=500, 
            log_exception=e
        )

def store_memory(user_message, assistant_response, memory_id='default'):
    timestamp = datetime.now().isoformat()
    
    # Store user message in existing memory system
    memory_manager.add_memory(
        f"User said: {user_message}",
        memory_id=memory_id,
        metadata={
            'timestamp': timestamp,
            'type': 'user_message'
        }
    )
    
    # Store assistant response in existing memory system
    memory_manager.add_memory(
        f"I responded: {assistant_response}",
        memory_id=memory_id,
        metadata={
            'timestamp': timestamp,
            'type': 'assistant_response'
        }
    )
    
    # Also store in Qdrant MCP if enabled
    if QDRANT_MCP_ENABLED:
        try:
            # Use the conversation store to store in Qdrant MCP
            success, error = conversation_store.store_conversation(
                user_message=user_message,
                assistant_response=assistant_response,
                conversation_id=memory_id,
                metadata={
                    'timestamp': timestamp,
                    'model': DEFAULT_MODEL
                }
            )
            
            if success:
                logger.info(f"Successfully stored conversation in Qdrant MCP")
            else:
                logger.warning(f"Failed to store conversation in Qdrant MCP: {error}")
        except Exception as e:
            logger.error(f"Error storing conversation in Qdrant MCP: {e}")
    
    logger.info(f"Stored memory: User and Assistant conversation in {memory_id}")

@app.route('/api/tags', methods=['GET'])
def ollama_tags_proxy():
    """Proxy Ollama tags API to support Docker container access."""
    try:
        logger.info(f"Proxying request to Ollama /api/tags endpoint at {ollama_host}")
        import requests
        
        # Use the properly configured host variable with explicit headers
        headers = {
            'Accept': 'application/json',
            'User-Agent': 'mem0-ollama/1.0'
        }
        
        # Make request with detailed logging
        logger.debug(f"Making GET request to {ollama_host}/api/tags")
        response = requests.get(f"{ollama_host}/api/tags", headers=headers, timeout=10)
        logger.debug(f"Received response with status code: {response.status_code}")

        if response.status_code == 200:
            try:
                # Log raw response for debugging
                raw_response = response.text
                logger.debug(f"Raw response from Ollama: {raw_response[:200]}...")
                
                data = response.json()
                models_count = len(data.get('models', []))
                logger.info(f"Ollama tags proxy success: {models_count} models found")
                
                # Return the json data with CORS headers
                return jsonify(data)
            except Exception as json_error:
                logger.error(f"Error parsing JSON from successful response: {json_error}")
                logger.error(f"Raw response content: {response.text[:500]}")
                return create_error_response(
                    message="Error parsing response from Ollama API", 
                    status_code=500, 
                    log_exception=json_error
                )
        else:
            error_content = response.text
            logger.error(f"Ollama returned non-200 status: {response.status_code}, Response: {error_content}")
            return create_error_response(
                message=f"Failed to reach Ollama API (Status: {response.status_code})", 
                status_code=502, 
                log_exception=f"Error response content: {error_content}"
            )
    except requests.exceptions.Timeout:
        logger.error(f"Timeout connecting to Ollama at {ollama_host}")
        return create_error_response(
            message="Connection to Ollama API timed out", 
            status_code=504, 
            log_exception="Request timed out after 10 seconds"
        )
    except requests.exceptions.ConnectionError as conn_err:
        logger.error(f"Connection error to Ollama at {ollama_host}: {conn_err}")
        return create_error_response(
            message="Could not connect to Ollama API", 
            status_code=503, 
            log_exception=f"Connection refused or failed: {str(conn_err)}"
        )
    except Exception as e:
        logger.error(f"Unexpected error in Ollama tags proxy: {e}")
        return create_error_response(
            message="Error connecting to Ollama API", 
            status_code=500, 
            log_exception=e
        )

@app.route('/api/pull', methods=['POST'])
def ollama_pull_proxy():
    """Proxy Ollama pull API to support Docker container access."""
    try:
        logger.info("Proxying request to Ollama /api/pull endpoint")
        import requests
        
        # Use the properly configured host variable
        data = request.json
        response = requests.post(f"{ollama_host}/api/pull", json=data, timeout=300)

        if response.status_code == 200:
            logger.info(f"Ollama pull proxy success")
            return jsonify(response.json())
        else:
            return create_error_response(
                message="Failed to reach Ollama API for model pull", 
                status_code=502, 
                log_exception=f"Status code: {response.status_code}"
            )
    except Exception as e:
        return create_error_response(
            message="Error connecting to Ollama API for model pull", 
            status_code=500, 
            log_exception=e
        )

@app.route('/api/health', methods=['GET'])
def health_check():
    """Get system health status."""
    try:
        logger.info("Health check requested")
        health_data = get_system_health()
        
        # Set appropriate status code based on health
        status_code = 200 if health_data["all_services_available"] else 503
        
        return jsonify(health_data), status_code
    except Exception as e:
        return create_error_response(
            message="Error performing health check", 
            status_code=500, 
            log_exception=e
        )

@app.route('/api/health/ollama', methods=['GET'])
def ollama_health():
    """Check Ollama service health."""
    try:
        status = check_ollama_health()
        status_code = 200 if status.available else 503
        return jsonify(status.to_dict()), status_code
    except Exception as e:
        return create_error_response(
            message="Error checking Ollama health", 
            status_code=500, 
            log_exception=e
        )

@app.route('/api/health/mem0', methods=['GET'])
def mem0_health():
    """Check mem0ai library health."""
    try:
        status = check_mem0_health()
        status_code = 200 if status.available else 503
        return jsonify(status.to_dict()), status_code
    except Exception as e:
        return create_error_response(
            message="Error checking mem0ai library health", 
            status_code=500, 
            log_exception=e
        )

@app.route('/api/health/qdrant', methods=['GET'])
def qdrant_health():
    """Check Qdrant vector database health."""
    try:
        status = check_qdrant_health()
        status_code = 200 if status.available else 503
        return jsonify(status.to_dict()), status_code
    except Exception as e:
        return create_error_response(
            message="Error checking Qdrant health", 
            status_code=500, 
            log_exception=e
        )
        
@app.route('/api/vector/status', methods=['GET'])
def vector_status():
    """Get detailed vector database status."""
    try:
        # Get info from mem0 adapter about vector database
        vector_db_info = memory_manager.get_memory_system_info()
        
        # Get Qdrant info separately 
        qdrant_status = check_qdrant_health()
        
        # Create combined response
        response = {
            "memory_system": vector_db_info,
            "qdrant": qdrant_status.to_dict(),
            "using_mem0ai": vector_db_info.get("current_mode") == "mem0ai",
            "using_qdrant": vector_db_info.get("current_mode") == "mem0ai" and qdrant_status.available,
            "embedding_dimensions": 4096  # From current configuration
        }
        
        return jsonify(response)
    except Exception as e:
        return create_error_response(
            message="Error getting vector database status", 
            status_code=500, 
            log_exception=e
        )

@app.route('/api/config/memory_system', methods=['GET', 'POST'])
def memory_system_config():
    """Get or set memory system configuration."""
    try:
        if request.method == 'GET':
            # Get current memory system info
            info = memory_manager.get_memory_system_info()
            return jsonify(info)
        else:
            # Set memory system mode
            try:
                data = request.json
                if not data or 'mode' not in data:
                    return jsonify({'error': 'No mode provided'}), 400
                    
                mode = data['mode']
                logger.info(f"Changing memory system mode to: {mode}")
                result = memory_manager.set_memory_mode(mode)
                
                if result.get('success', False):
                    logger.info(f"Memory system mode changed successfully: {result}")
                    return jsonify(result)
                else:
                    logger.error(f"Error changing memory system mode: {result}")
                    return jsonify({'error': result.get('error', 'Unknown error')}), 400
            except Exception as e:
                logger.error(f"Exception in memory system config POST: {e}")
                return jsonify({'error': str(e)}), 400
    except Exception as e:
        logger.error(f"Exception in memory system config: {e}")
        return jsonify({'error': str(e)}), 500

# Create a standardized error response function
def create_error_response(message, status_code=500, log_exception=None):
    """Create a standardized error response with appropriate logging
    
    Args:
        message: User-facing error message (keep it generic for security)
        status_code: HTTP status code to return
        log_exception: Exception to log (not exposed to user)
    """
    if log_exception:
        logger.error(f"Error: {message} - Details: {str(log_exception)}")
    else:
        logger.error(f"Error: {message}")
        
    # Only return detailed error information in debug mode
    if DEBUG:
        detail = str(log_exception) if log_exception else None
        return jsonify({"error": message, "detail": detail}), status_code
    else:
        return jsonify({"error": message}), status_code

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=PORT, debug=DEBUG)
