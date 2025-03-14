from flask import Flask
import time

app = Flask(__name__)

@app.route('/')
def hello():
    return "Flask test working!"

if __name__ == '__main__':
    print("Starting test Flask server on port 8080")
    # Try with threaded=False to see if there's a threading issue
    app.run(host='0.0.0.0', port=8080, debug=False, threaded=False)
    print("This line should only print after server shutdown")
