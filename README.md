# mem0-Ollama

Web chat interface with mem0 integration for Ollama

## Overview

mem0-Ollama is a web-based chat interface that integrates mem0 for memory management with Ollama for local LLM inference. This project enables context-aware conversations with local models through a responsive web UI.

## Features

- **Memory Management**: Context-aware conversations using mem0 vector storage
- **Active/Inactive Memory States**: Dynamic memory tracking for relevant context
- **Web Interface**: Responsive chat UI with memory visualization
- **Ollama Integration**: Works with any model available in Ollama
- **Docker Support**: Container-ready deployment with proper networking

## Requirements

- Python 3.9+
- Flask and required Python packages
- Ollama running locally or in a container
- mem0 vector database

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/KhryptorGraphics/mem0-ollama.git
   cd mem0-ollama
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Make sure Ollama is running locally or adjust the `OLLAMA_HOST` in config.py

4. Run the application:
   ```
   python main.py
   ```

5. Open your browser and navigate to http://localhost:5000

## Docker Deployment

You can also run the application with Docker:

```
docker-compose up -d
```

This will start both the mem0-ollama service and the mem0 vector database.

## Configuration

Adjust settings in `config.py` to customize:

- Ollama host URL
- Default model
- System prompt
- Memory settings
- Server port

## License

See the [LICENSE](LICENSE) file for details.
