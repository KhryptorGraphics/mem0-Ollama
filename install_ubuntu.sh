#!/bin/bash

# Installation script for mem0-Ollama on Ubuntu 24.04
set -e

echo "=== mem0-Ollama Installation Script for Ubuntu 24.04 ==="
echo "This script will install mem0-Ollama and its dependencies."
echo 

# Check if running as root
if [ "$(id -u)" -eq 0 ]; then
    echo "Please run this script as a regular user, not as root."
    exit 1
fi

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Install system dependencies
echo "Installing system dependencies..."
sudo apt update
sudo apt install -y python3 python3-pip python3-venv git curl wget

# Check for Docker installation
if ! command_exists docker; then
    echo "Installing Docker..."
    curl -fsSL https://get.docker.com -o get-docker.sh
    sudo sh get-docker.sh
    sudo usermod -aG docker $USER
    rm get-docker.sh
    echo "Docker installed. You may need to log out and log back in for group changes to take effect."
else
    echo "Docker is already installed."
fi

# Check for Docker Compose installation
if ! command_exists docker-compose; then
    echo "Installing Docker Compose..."
    sudo apt install -y docker-compose-plugin
else
    echo "Docker Compose is already installed."
fi

# Check for Ollama installation
if ! command_exists ollama; then
    echo "Installing Ollama..."
    curl -fsSL https://ollama.com/install.sh | sh
else
    echo "Ollama is already installed."
fi

# Create a directory for the project
PROJECT_DIR="$HOME/mem0-ollama"
if [ ! -d "$PROJECT_DIR" ]; then
    echo "Creating project directory at $PROJECT_DIR"
    mkdir -p "$PROJECT_DIR"
fi

# Clone the repository
echo "Cloning mem0-Ollama repository..."
if [ -d "$PROJECT_DIR/.git" ]; then
    echo "Repository already exists. Pulling latest changes..."
    cd "$PROJECT_DIR"
    git pull
else
    git clone https://github.com/KhryptorGraphics/mem0-ollama.git "$PROJECT_DIR"
    cd "$PROJECT_DIR"
fi

# Create a virtual environment
echo "Setting up Python virtual environment..."
if [ ! -d "$PROJECT_DIR/venv" ]; then
    python3 -m venv "$PROJECT_DIR/venv"
fi

# Activate the virtual environment
source "$PROJECT_DIR/venv/bin/activate"

# Install Python dependencies
echo "Installing Python dependencies..."
pip install -r "$PROJECT_DIR/requirements.txt"

# Create a startup script
echo "Creating startup script..."
cat > "$PROJECT_DIR/start.sh" << 'EOL'
#!/bin/bash
cd "$(dirname "$0")"
source venv/bin/activate
python main.py
EOL

chmod +x "$PROJECT_DIR/start.sh"

# Create a Docker startup script
echo "Creating Docker startup script..."
cat > "$PROJECT_DIR/start_docker.sh" << 'EOL'
#!/bin/bash
cd "$(dirname "$0")"
docker-compose up -d
EOL

chmod +x "$PROJECT_DIR/start_docker.sh"

echo 
echo "=== Installation Complete! ==="
echo 
echo "To start mem0-Ollama:"
echo "1. Direct mode: $PROJECT_DIR/start.sh"
echo "2. Docker mode: $PROJECT_DIR/start_docker.sh"
echo 
echo "Then open your browser at http://localhost:5000"
echo 
echo "Note: If using Docker, make sure Ollama is running before starting the container."
echo "You can start Ollama with: ollama serve"
echo 
