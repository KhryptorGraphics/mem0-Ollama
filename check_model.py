#!/usr/bin/env python3
"""
Model Check Utility for mem0-ollama

This script verifies that the required model is available in Ollama and
pulls it if necessary. It helps diagnose and fix model-related issues.
"""

import argparse
import sys
import time
from typing import Optional

from config import OLLAMA_HOST, DEFAULT_MODEL
from ollama_client import OllamaClient
import logging_utils

# Set up logger
logger = logging_utils.get_logger('model_check')

def check_and_pull_model(model: str, client: OllamaClient, auto_pull: bool = False) -> bool:
    """
    Check if a model is available and pull it if needed and auto_pull is enabled.
    
    Args:
        model: Model name to check
        client: OllamaClient instance
        auto_pull: Whether to automatically pull the model if not available
        
    Returns:
        True if model is or becomes available, False otherwise
    """
    print(f"Checking if model '{model}' is available...")
    
    # First, check if model is already available
    if client.verify_model_available(model):
        print(f"✅ Model '{model}' is already available")
        return True
    
    # Model not available
    print(f"❌ Model '{model}' is not available")
    available_models = client.get_available_models()
    if available_models:
        print(f"Available models: {', '.join(available_models)}")
    
    # If auto_pull is enabled, try to pull the model
    if auto_pull:
        print(f"Attempting to pull model '{model}' (this may take a while)...")
        success = client.pull_model(model)
        if success:
            print(f"✅ Successfully pulled model '{model}'")
            return True
        else:
            print(f"❌ Failed to pull model '{model}'")
            return False
    else:
        print(f"\nTo pull this model, run: ollama pull {model}")
        print(f"Or restart this script with --auto-pull flag")
        return False

def diagnose_ollama_issues(client: OllamaClient):
    """Provide troubleshooting guidance for Ollama issues"""
    print("\n=== Ollama Troubleshooting ===")
    print("1. Make sure Ollama is installed: https://ollama.com/download")
    print("2. Check if Ollama is running (it should start at boot by default)")
    print("3. On Windows, check the Ollama icon in the system tray")
    print("4. On macOS/Linux, try: pkill ollama && ollama serve")
    print(f"5. Verify the Ollama API is accessible at: {client.host}")
    print("6. Try restarting your computer")
    print("7. Check firewall settings if using a remote Ollama server")

def main():
    parser = argparse.ArgumentParser(description="Check and pull Ollama models")
    parser.add_argument("--model", help=f"Model to check (default: {DEFAULT_MODEL})",
                       default=DEFAULT_MODEL)
    parser.add_argument("--host", help=f"Ollama host (default: {OLLAMA_HOST})",
                       default=OLLAMA_HOST)
    parser.add_argument("--auto-pull", action="store_true", 
                       help="Automatically pull model if not available")
    args = parser.parse_args()
    
    # Print header
    print("=" * 70)
    print(f"mem0-ollama Model Check Utility")
    print("=" * 70)
    print(f"Ollama Host: {args.host}")
    print(f"Target Model: {args.model}")
    print("-" * 70)
    
    # Create client
    client = OllamaClient(host=args.host)
    
    # First, check if Ollama is running
    print("Checking Ollama server status...")
    if not client.is_server_running():
        print("❌ Ollama server is not running or not accessible")
        diagnose_ollama_issues(client)
        return 1
    
    print("✅ Ollama server is running")
    
    # Check available models
    print("\nFetching available models...")
    models = client.get_available_models(force_refresh=True)
    
    if not models:
        print("⚠️ No models found in Ollama")
    else:
        print(f"Found {len(models)} models: {', '.join(models)}")
    
    # Check and pull the specified model
    print("\n" + "-" * 70)
    success = check_and_pull_model(args.model, client, args.auto_pull)
    
    # Provide a summary
    print("\n" + "=" * 70)
    if success:
        print(f"✅ SUCCESS: Model '{args.model}' is available")
        print("You can start the mem0-ollama application now")
        return 0
    else:
        print(f"❌ FAILURE: Model '{args.model}' is not available")
        print("Please pull the model before starting mem0-ollama")
        return 1

if __name__ == "__main__":
    sys.exit(main())
