from flask import Flask

app = Flask(__name__)

@app.route('/')
def hello():
    return "Flask test on port 8000 is working!"

if __name__ == '__main__':
    print("Starting test Flask server on port 8000")
    try:
        app.run(host='0.0.0.0', port=8000, debug=False, threaded=False)
    except Exception as e:
        print(f"Error starting server: {e}")
