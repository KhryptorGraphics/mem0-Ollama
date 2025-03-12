from api import app
from config import PORT, DEBUG

if __name__ == '__main__':
    print(f"Starting mem0-Ollama server on port {PORT}")
    print(f"Access web interface at http://localhost:{PORT}/")
    app.run(host='0.0.0.0', port=PORT, debug=DEBUG)
