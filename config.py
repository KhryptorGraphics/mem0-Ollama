import os

# Ollama Configuration
OLLAMA_HOST = os.environ.get('OLLAMA_HOST', 'http://localhost:11434')

# Default Model Settings
DEFAULT_MODEL = os.environ.get('DEFAULT_MODEL', 'llama3')
DEFAULT_SYSTEM_PROMPT = os.environ.get('DEFAULT_SYSTEM_PROMPT', '''
You are a helpful assistant with access to memory. When relevant information from memory is provided, use it to inform your responses.
''')

# Memory Settings
MEM0_HOST = os.environ.get('MEM0_HOST', 'http://localhost:8000')
MAX_MEMORIES = int(os.environ.get('MAX_MEMORIES', 5))
MEMORY_DATABASE = os.environ.get('MEMORY_DATABASE', 'mem0_ollama.json')

# Server Configuration
PORT = int(os.environ.get('PORT', 5000))

# Debug Settings
DEBUG = os.environ.get('DEBUG', 'False').lower() == 'true'

# Collection Names
VECTOR_COLLECTION = os.environ.get('VECTOR_COLLECTION', 'memories')
ACTIVE_MEMORY_COLLECTION = os.environ.get('ACTIVE_MEMORY_COLLECTION', 'active_memories')
