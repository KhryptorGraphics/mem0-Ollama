# Mem0-Ollama Connection Guide

This guide helps you set up and troubleshoot the connection between Mem0-Ollama and Ollama server.

## Prerequisites

1. [Install Ollama](https://ollama.com/download) for your operating system
2. Make sure Ollama is running before starting Mem0-Ollama

## Quick Start

1. **Check Ollama installation**: Make sure Ollama is installed and running
2. **Check available models**: Run `python check_model.py` to verify Ollama is accessible
3. **Start the application**: Run `python main.py` to start the Mem0-Ollama server
4. **Access the interface**: Open http://localhost:8000 in your browser

## Troubleshooting Connection Issues

If you're having trouble connecting to Ollama, try these steps:

### Step 1: Verify Ollama is Running

Check if the Ollama service is running:

```bash
python check_model.py
```

This utility will check if Ollama is running and accessible, and will list the available models.

### Step 2: Pull Required Models

If you don't have the required models, you can pull them with:

```bash
# Using Ollama CLI (recommended)
ollama pull llama3:latest

# Or using our utility
python check_model.py --model llama3:latest --auto-pull
```

### Step 3: Check Host Configuration

By default, Mem0-Ollama connects to Ollama at `http://localhost:11434`. If your Ollama is running on a different host or port, you can specify it with:

```bash
# Set an environment variable
set OLLAMA_HOST=http://your-ollama-host:port  # Windows
export OLLAMA_HOST=http://your-ollama-host:port  # Linux/macOS

# Or pass it explicitly when running the check utility
python check_model.py --host http://your-ollama-host:port
```

### Step 4: Test Direct Connection

You can test a direct connection to Ollama with:

```bash
python direct_ollama_server.py
```

Then open http://localhost:8000/direct in your browser.

### Common Issues

1. **"Connection failed" error in the UI**
   - Make sure Ollama is running
   - Check that the Ollama host is correctly set to `http://localhost:11434`
   - Restart the Mem0-Ollama server

2. **Models not showing up**
   - Check if models are available with `ollama list` in terminal
   - If models are missing, pull them with `ollama pull model-name`
   - Click the refresh button next to the model dropdown in the UI

3. **Error "Failed to load resource" in browser console**
   - This may indicate a CORS issue or a server-side error
   - Check the terminal running the server for error messages
   - Try a different browser or clear browser cache

## Advanced Configuration

### Environment Variables

- `OLLAMA_HOST`: URL for the Ollama server (default: `http://localhost:11434`)
- `PORT`: Port for the Mem0-Ollama server (default: `8000`)
- `DEFAULT_MODEL`: Default model to use (default: `llama3:latest`)
- `DEBUG`: Enable debug mode (default: `False`)

### Memory Settings

- `MEM0_HOST`: URL for the mem0 vector database (default: `http://localhost:6333`)
- `MAX_MEMORIES`: Maximum number of memories to retrieve (default: `5`)
- `MEMORY_DATABASE`: File to store memories (default: `mem0_ollama.json`)

## Project Structure

- `main.py`: Main application entry point
- `api.py`: API endpoints for the application
- `ollama_client.py`: Client for interacting with Ollama API
- `memory_utils.py`: Utilities for memory management
- `chat_interface.html`: Web UI for chat interface
- `check_model.py`: Utility to check Ollama model availability
- `health_check.py`: Health check endpoints for monitoring
