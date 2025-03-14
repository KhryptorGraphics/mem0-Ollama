# Mem0 Adapter for Ollama

A robust adapter for integrating mem0ai with Ollama, providing persistent memory capabilities with automatic fallback support.

## Overview

This adapter provides a bridge between [mem0ai](https://github.com/mem0ai) and [Ollama](https://ollama.ai/), allowing applications to leverage mem0ai's powerful memory systems with Ollama's local LLM capabilities. It automatically handles common integration issues and provides a consistent API regardless of the underlying implementation.

## Features

- **Dual-mode Memory System**: 
  - Full mem0ai integration with Ollama and Qdrant when available
  - Automatic fallback to JSON-based storage when mem0ai fails or isn't available
  - Graceful error handling with automatic recovery

- **Automatic Model Detection**:
  - Uses dedicated embedding models like `nomic-embed-text` when available
  - Falls back to general models when needed
  - Configures embedding dimensions correctly

- **Robust Vector Store Integration**:
  - Fixes vector dimension mismatch problems with automatic collection reset
  - Handles Qdrant initialization properly
  - Ensures consistent memory object format regardless of backend

- **Smart Error Recovery**:
  - Auto-detects and handles common errors like vector dimension mismatches
  - Preserves existing memories during failover
  - Maintains consistent API across different modes

## Requirements

- Python 3.8+
- Ollama server running (default: http://localhost:11434)
- Qdrant vector database (default: http://localhost:6333)
- mem0ai library (optional, falls back to built-in implementation if not available)

## Installation

1. Copy the `final_mem0_adapter.py` file to your project directory.

2. Configure your project's dependencies to include:
```
ollama
requests
```

3. Optional but recommended for full functionality:
```
mem0ai
qdrant-client
```

## Usage

### Basic Usage

```python
from final_mem0_adapter import Mem0Adapter

# Create adapter with default settings
memory = Mem0Adapter()

# Add a memory
memory_id = memory.add_memory("This is a fact to remember", "user_1234")

# Retrieve relevant memories
results = memory.get_relevant_memories("What fact should I remember?", "user_1234")
for memory in results:
    print(f"Memory: {memory['text']}, Score: {memory['score']}")

# List all memories
all_memories = memory.list_memories("user_1234")
print(f"Found {len(all_memories)} memories")
```

### Customizing Configuration

```python
# Custom configuration
adapter = Mem0Adapter(
    ollama_host="http://localhost:11434",
    mem0_host="http://localhost:6333",
    default_model="llama3:latest",
    memory_mode="auto"  # or "mem0ai" or "fallback"
)
```

### Memory Modes

The adapter supports three modes:

- **auto**: Tries to use mem0ai with graceful fallback (recommended)
- **mem0ai**: Forces use of mem0ai (fails if not available)
- **fallback**: Forces use of the built-in fallback implementation

```python
# Change memory mode at runtime
result = adapter.set_memory_mode(adapter.MEMORY_MODE_FALLBACK)
print(f"Mode change success: {result['success']}")
```

### Working with Memory Groups

```python
# Add memories to different user groups
adapter.add_memory("Alice's memory", "user_alice")
adapter.add_memory("Bob's memory", "user_bob")

# Retrieve specific group memories
alice_memories = adapter.list_memories("user_alice")
bob_memories = adapter.list_memories("user_bob")
```

### Active Memory Tracking

The adapter keeps track of which memories are active in the current conversation:

```python
# Get active memories in the current context
active_ids = adapter.get_active_memories("user_1234")

# Reset active memories
adapter.reset_active_memories("user_1234")
```

## Error Handling

The adapter provides robust error handling and automatically recovers from common issues:

```python
try:
    memory_id = adapter.add_memory("Memory that might fail", "user_1234")
    if not memory_id:
        print("Memory addition failed but adapter handled it gracefully")
except Exception as e:
    print(f"Unexpected error: {e}")
```

## System Information

Get information about the current memory system:

```python
info = adapter.get_memory_system_info()
print(f"Current mode: {info['current_mode']}")
print(f"Using fallback: {info['using_fallback']}")
print(f"Memory file: {info['memory_file']}")
print(f"Vector DB: {info['vector_db']}")
```

## Integration with Ollama

When using this adapter with Ollama, ensure:

1. Ollama is running and accessible at the specified host URL
2. Appropriate models are installed in Ollama (especially embedding models)
3. Qdrant is properly configured and accessible

## Troubleshooting

1. **Vector Dimension Mismatch**: If you see vector dimension errors, the adapter will automatically handle them by falling back to the simpler store.

2. **Memory Storage Issues**: If you encounter issues with memory storage, check:
   - Qdrant connection settings
   - Memory file permissions (for fallback store)
   - Proper model compatibility

3. **Connection Issues**: If unable to connect to Ollama or Qdrant:
   - Verify services are running
   - Check network settings
   - Ensure firewall allows connections

## Advanced Usage

### Custom Metadata

You can attach custom metadata to memories:

```python
metadata = {
    "source": "user_input",
    "timestamp": "2025-03-13T22:30:00",
    "importance": "high"
}

memory_id = adapter.add_memory("Important fact", "user_1234", metadata=metadata)
```

### Memory Filtering

Filter memories by source or other metadata:

```python
# List only memories from a specific source
user_memories = adapter.list_memories("user_1234", source="user_input")
```

## Testing

Run the included test script to verify the adapter is working properly:

```bash
python test_final_adapter.py
```

## License

Same as the original project.
