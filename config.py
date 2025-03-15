import os

# Qdrant MCP Configuration
QDRANT_MCP_ENABLED = os.environ.get('QDRANT_MCP_ENABLED', 'True').lower() == 'true'

# Ollama Configuration
OLLAMA_HOST = os.environ.get('OLLAMA_HOST', 'http://localhost:11434')
# Force localhost if using 0.0.0.0 (common misconfiguration)
if '0.0.0.0' in OLLAMA_HOST:
    OLLAMA_HOST = 'http://localhost:11434'
# Override default if we're in Docker or need explicit host setting
if os.environ.get('DOCKER_ENV', 'False').lower() == 'true':
    # Use Docker network address for Ollama in container setup
    OLLAMA_HOST = os.environ.get('OLLAMA_HOST', 'http://ollama:11434')

# Default Model Settings
DEFAULT_MODEL = os.environ.get('DEFAULT_MODEL', 'llama3:latest')
DEFAULT_SYSTEM_PROMPT = os.environ.get('DEFAULT_SYSTEM_PROMPT', '''
You are a helpful assistant with access to memory. When relevant information from memory is provided, use it to inform your responses.
''')

# Memory Settings
# If using Docker, the Qdrant container is available at localhost:6333
# We should connect to it for mem0ai integration
MEM0_HOST = os.environ.get('MEM0_HOST', 'http://localhost:6333')
MAX_MEMORIES = int(os.environ.get('MAX_MEMORIES', 5))
MEMORY_DATABASE = os.environ.get('MEMORY_DATABASE', 'mem0_ollama.json')

# Server Configuration
PORT = int(os.environ.get('PORT', 8000))  # Default port for API and web interface

# Debug Settings
DEBUG = os.environ.get('DEBUG', 'False').lower() == 'true'

# Security Settings
ALLOWED_ORIGINS = os.environ.get('ALLOWED_ORIGINS', 'http://localhost:8000,http://127.0.0.1:8000')

# Collection Names
VECTOR_COLLECTION = os.environ.get('VECTOR_COLLECTION', 'memories')
ACTIVE_MEMORY_COLLECTION = os.environ.get('ACTIVE_MEMORY_COLLECTION', 'active_memories')
