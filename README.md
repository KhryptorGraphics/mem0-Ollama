# mem0-Ollama

![mem0-Ollama](https://raw.githubusercontent.com/KhryptorGraphics/mem0-ollama/main/docs/logo.png)

> Web chat interface with mem0 integration for Ollama

## Overview

mem0-Ollama is a web-based chat interface that integrates mem0 for memory management with Ollama for local LLM inference. This project enables context-aware conversations with local models through a responsive web UI.

## Features

- **Memory Management**: Context-aware conversations using mem0 vector storage
- **Active/Inactive Memory States**: Dynamic memory tracking for relevant context
- **Web Interface**: Responsive chat UI with memory visualization
- **Ollama Integration**: Works with any model available in Ollama
- **Docker Support**: Container-ready deployment with proper networking

## Installation

### Windows

1. Download the installation script:
   ```
   curl -o install_windows.ps1 https://raw.githubusercontent.com/KhryptorGraphics/mem0-ollama/main/install_windows.ps1
   ```

2. Run the script as Administrator:
   ```
   powershell -ExecutionPolicy Bypass -File install_windows.ps1
   ```

3. Follow the on-screen instructions to complete the installation.

### Ubuntu 24.04

1. Download the installation script:
   ```
   curl -o install_ubuntu.sh https://raw.githubusercontent.com/KhryptorGraphics/mem0-ollama/main/install_ubuntu.sh
   ```

2. Make the script executable:
   ```
   chmod +x install_ubuntu.sh
   ```

3. Run the script:
   ```
   ./install_ubuntu.sh
   ```

4. Follow the on-screen instructions to complete the installation.

### Manual Installation

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

5. Open your browser and navigate to http://localhost:8000

## Docker Deployment

You can also run the application with Docker:

```
docker-compose up -d
```

This will start both the mem0-ollama service and the mem0 vector database.

## Usage

### Web Interface

The web interface provides two modes:

1. **Memory-enabled Chat**: Access at http://localhost:8000/
   - Features memory management for context-aware conversations
   - Visualizes active memories being used for context
   - Maintains conversation history with memory persistence

2. **Direct Ollama Chat**: Access at http://localhost:8000/direct
   - Direct communication with Ollama without mem0 integration
   - Simpler interface for direct model testing
   - No memory persistence between conversations

### Configuration

Adjust settings in `config.py` to customize:

- Ollama host URL
- Default model
- System prompt
- Memory settings
- Server port

## Requirements

- Python 3.9+
- Flask and required Python packages
- Ollama running locally or in a container
- mem0 vector database (included in Docker setup)

## Contributing

We welcome contributions! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
