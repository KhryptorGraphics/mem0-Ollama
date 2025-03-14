#!/usr/bin/env python3
"""
Ollama Connection Test Script

This script tests the connection to the Ollama API server and verifies
that the required endpoints are available and working correctly.
"""

import argparse
import json
import os
import requests
import sys
import time
from typing import Dict, Any, List, Optional, Tuple

DEFAULT_HOST = "http://localhost:11434"
TEST_PROMPT = "Hello, can you respond with a short greeting to confirm you're working?"
DEFAULT_MODEL = "llama3"
TEST_TIMEOUT = 15  # seconds

def format_json(data: Dict[str, Any]) -> str:
    """Format JSON data for display with indentation."""
    return json.dumps(data, indent=2)

def check_ollama_running(host: str) -> Tuple[bool, str]:
    """Check if Ollama server is running at the given host."""
    try:
        response = requests.get(f"{host}/api/version", timeout=5)
        if response.status_code == 200:
            version_info = response.json()
            return True, f"Ollama is running (version: {version_info.get('version', 'unknown')})"
        else:
            return False, f"Ollama server responded with status code {response.status_code}"
    except requests.exceptions.ConnectionError:
        return False, f"Could not connect to Ollama at {host}. Is the server running?"
    except requests.exceptions.Timeout:
        return False, f"Connection to Ollama at {host} timed out"
    except Exception as e:
        return False, f"Error checking Ollama: {str(e)}"

def list_available_models(host: str) -> Tuple[bool, str, List[str]]:
    """Get a list of available models from Ollama."""
    try:
        response = requests.get(f"{host}/api/tags", timeout=5)
        if response.status_code == 200:
            data = response.json()
            models = data.get('models', [])
            model_names = [model.get('name', '') for model in models]
            
            if model_names:
                message = f"Found {len(model_names)} available models"
                return True, message, model_names
            else:
                return False, "No models found in Ollama", []
        else:
            return False, f"Failed to list models: HTTP {response.status_code}", []
    except Exception as e:
        return False, f"Error listing models: {str(e)}", []

def check_model_availability(host: str, model_name: str) -> Tuple[bool, str]:
    """Check if a specific model is available."""
    success, message, models = list_available_models(host)
    if not success:
        return False, message
    
    if model_name in models:
        return True, f"Model '{model_name}' is available"
    else:
        available = ", ".join(models) if models else "No models available"
        return False, f"Model '{model_name}' not found. Available models: {available}"

def test_generate_endpoint(host: str, model_name: str) -> Tuple[bool, str]:
    """Test the generate endpoint with a simple prompt."""
    try:
        url = f"{host}/api/generate"
        payload = {
            "model": model_name,
            "prompt": TEST_PROMPT,
            "stream": False
        }
        
        print(f"Testing /api/generate with model '{model_name}'...")
        start_time = time.time()
        response = requests.post(url, json=payload, timeout=TEST_TIMEOUT)
        elapsed = time.time() - start_time
        
        if response.status_code == 200:
            data = response.json()
            if 'response' in data:
                return True, f"Generated response in {elapsed:.2f}s: '{data['response'][:50]}...'"
            else:
                return False, f"Response doesn't contain expected 'response' field: {format_json(data)}"
        else:
            error_text = response.text
            return False, f"Generate endpoint failed: HTTP {response.status_code} - {error_text}"
    except Exception as e:
        return False, f"Error testing generate endpoint: {str(e)}"

def run_all_tests(host: str, model: str) -> bool:
    """Run all connection tests and return overall success status."""
    # Ensure host has proper URL scheme
    if not host.startswith(('http://', 'https://')):
        host = f"http://{host}"
    
    print(f"\n=== Ollama Connection Tests ===")
    print(f"Host: {host}")
    print(f"Test model: {model}")
    print("=" * 30)
    
    # Test 1: Check if Ollama is running
    success, message = check_ollama_running(host)
    print(f"\n1. Server Check: {'✅ PASS' if success else '❌ FAIL'}")
    print(f"   {message}")
    
    if not success:
        print("\n⚠️  Ollama server is not reachable. Further tests will be skipped.")
        print("\nTroubleshooting tips:")
        print("1. Ensure Ollama is installed: https://ollama.com/download")
        print("2. Make sure Ollama server is running")
        print("3. Check firewall settings if running on a remote host")
        print("4. Verify the correct host address (default: http://localhost:11434)")
        return False
    
    # Test 2: List available models
    success, message, models = list_available_models(host)
    print(f"\n2. Model Listing: {'✅ PASS' if success else '❌ FAIL'}")
    print(f"   {message}")
    
    if success and models:
        print("   Available models:")
        for m in models:
            print(f"   - {m}")
    
    # Test 3: Check if test model is available
    success, message = check_model_availability(host, model)
    print(f"\n3. Model Availability: {'✅ PASS' if success else '❌ FAIL'}")
    print(f"   {message}")
    
    # Test 4: Test generate endpoint
    if success:
        success, message = test_generate_endpoint(host, model)
        print(f"\n4. Generation Test: {'✅ PASS' if success else '❌ FAIL'}")
        print(f"   {message}")
    else:
        print("\n4. Generation Test: ⚠️  SKIPPED (model not available)")
        print("   To pull a model, use: 'ollama pull MODEL_NAME'")
    
    print("\n=== Test Summary ===")
    if success:
        print("✅ All tests passed successfully!")
        print(f"Ollama is running correctly at {host}")
        return True
    else:
        print("❌ Some tests failed. See detailed messages above.")
        return False

def main():
    parser = argparse.ArgumentParser(description="Test connection to Ollama API")
    parser.add_argument("--host", default=DEFAULT_HOST, help=f"Ollama host URL (default: {DEFAULT_HOST})")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"Model to test (default: {DEFAULT_MODEL})")
    args = parser.parse_args()
    
    success = run_all_tests(args.host, args.model)
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
