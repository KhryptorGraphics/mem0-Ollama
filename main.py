#!/usr/bin/env python3
"""
Main entry point for mem0 + Ollama integration

This script starts the web interface and API server for the mem0 + Ollama integration.
"""

import os
import sys
import time
import argparse
import logging
import traceback
from typing import Optional, Tuple

from config import OLLAMA_HOST, OLLAMA_MODEL, QDRANT_HOST, API_PORT
from ollama_client import check_ollama
from memory_utils import check_qdrant, initialize_memory
from api import run_server
import logging_utils

# Get a logger for the main module
logger = logging_utils.get_logger(__name__)

# Capture start time for performance tracking
start_time = time.time()

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Start the mem0 + Ollama integration server")
    
    # Service configuration
    parser.add_argument(
        "--ollama-host", 
        type=str, 
        default=OLLAMA_HOST,
        help=f"Ollama API host (default: {OLLAMA_HOST})"
    )
    
    parser.add_argument(
        "--ollama-model", 
        type=str, 
        default=OLLAMA_MODEL,
        help=f"Ollama model to use (default: {OLLAMA_MODEL})"
    )
    
    parser.add_argument(
        "--qdrant-host", 
        type=str, 
        default=QDRANT_HOST,
        help=f"Qdrant host (default: {QDRANT_HOST})"
    )
    
    parser.add_argument(
        "--embed-model",
        type=str,
        default=None,
        help="Model to use for embeddings (defaults to ollama-model if not specified)"
    )
    
    parser.add_argument(
        "--port",
        type=int,
        default=API_PORT,
        help=f"Port for the API server (default: {API_PORT})"
    )
    
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Run in debug mode"
    )
    
    # Logging configuration
    parser.add_argument(
        "--log-level",
        type=str,
        choices=["debug", "info", "warning", "error", "critical"],
        default="info",
        help="Set the logging level (default: info)"
    )
    
    parser.add_argument(
        "--log-to-console",
        action="store_true",
        default=True,
        help="Log to console (default: True)"
    )
    
    parser.add_argument(
        "--log-to-file",
        action="store_true",
        default=True,
        help="Log to files (default: True)"
    )
    
    parser.add_argument(
        "--log-dir",
        type=str,
        default="logs",
        help="Directory for log files (default: logs)"
    )
    
    parser.add_argument(
        "--max-log-size",
        type=int,
        default=10 * 1024 * 1024,  # 10 MB
        help="Maximum log file size in bytes before rotation (default: 10MB)"
    )
    
    parser.add_argument(
        "--backup-count",
        type=int,
        default=10,
        help="Number of backup log files to keep (default: 10)"
    )
    
    return parser.parse_args()

def setup_logging(args):
    """
    Set up logging based on command line arguments.
    
    Args:
        args: Parsed command line arguments
    """
    # Create logs directory if it doesn't exist
    os.makedirs(args.log_dir, exist_ok=True)
    
    # Update the log directory in logging_utils
    logging_utils.LOG_DIR = args.log_dir
    logging_utils.MAIN_LOG = os.path.join(args.log_dir, "main.log")
    logging_utils.ERROR_LOG = os.path.join(args.log_dir, "error.log")
    logging_utils.API_LOG = os.path.join(args.log_dir, "api_calls.log")
    logging_utils.MEMORY_LOG = os.path.join(args.log_dir, "memory_ops.log")
    
    # Configure logging
    logging_utils.configure_logging(
        level=args.log_level,
        console=args.log_to_console,
        file=args.log_to_file,
        max_file_size=args.max_log_size,
        backup_count=args.backup_count
    )
    
    # Reduce Flask logs
    werkzeug_logger = logging.getLogger('werkzeug')
    werkzeug_logger.setLevel(logging.WARNING)
    
    logger.info(f"Logging configured: level={args.log_level}, console={args.log_to_console}, file={args.log_to_file}")
    logger.info(f"Log directory: {args.log_dir}")

def main():
    """Main entry point."""
    args = parse_args()
    
    # Set up logging first
    setup_logging(args)
    
    # Generate a request ID for the startup process
    request_id = logging_utils.set_request_id()
    logger.info(f"Starting mem0 + Ollama integration (request_id={request_id})")
    
    # Override global settings with command line arguments
    if args.ollama_host != OLLAMA_HOST:
        logger.info(f"Using custom Ollama host: {args.ollama_host}")
        os.environ["OLLAMA_HOST"] = args.ollama_host
    
    if args.ollama_model != OLLAMA_MODEL:
        logger.info(f"Using custom Ollama model: {args.ollama_model}")
        os.environ["OLLAMA_MODEL"] = args.ollama_model
    
    if args.qdrant_host != QDRANT_HOST:
        logger.info(f"Using custom Qdrant host: {args.qdrant_host}")
        os.environ["QDRANT_HOST"] = args.qdrant_host
    
    # Log system information
    try:
        import platform
        sys_info = {
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "processor": platform.processor(),
            "machine": platform.machine()
        }
        logger.info(f"System information: {sys_info}")
    except Exception as e:
        logger.warning(f"Could not get system information: {e}")
    
    # Check Ollama and Qdrant connections
    logger.info("Checking Ollama connection...")
    ollama_ok, ollama_msg = check_ollama()
    if not ollama_ok:
        logger.error(f"Ollama check failed: {ollama_msg}")
        logger.warning("Continuing despite Ollama connection failure")
    else:
        logger.info(f"Ollama check successful: {ollama_msg}")
    
    logger.info("Checking Qdrant connection...")
    qdrant_ok, qdrant_msg = check_qdrant()
    if not qdrant_ok:
        logger.error(f"Qdrant check failed: {qdrant_msg}")
        logger.warning("Continuing despite Qdrant connection failure")
    else:
        logger.info(f"Qdrant check successful: {qdrant_msg}")
    
    # Initialize memory with proper error handling
    logger.info("Initializing memory system...")
    memory = None
    try:
        memory = initialize_memory(
            ollama_model=args.ollama_model,
            embed_model=args.embed_model
        )
        logger.info("Memory system initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize memory: {e}")
        logging_utils.log_exception(logger, e, {
            "ollama_model": args.ollama_model,
            "embed_model": args.embed_model
        })
        logger.info("Memory system will be initialized when first needed")
    
    # Start the API server with improved error handling
    logger.info(f"Starting API server on port {args.port}...")
    server_start_time = time.time()
    
    # Try with primary port
    success = False
    try:
        run_server(port=args.port, debug=args.debug)
        success = True
    except OSError as e:
        server_error = e
        logger.error(f"Failed to start server on port {args.port}: {str(e)}")
        logging_utils.log_exception(logger, e, {"port": args.port})
        
        # Try alternate port
        alternate_port = args.port + 1
        logger.info(f"Trying alternate port {alternate_port}...")
        
        try:
            run_server(port=alternate_port, debug=args.debug)
            logger.info(f"Server started successfully on alternate port {alternate_port}")
            success = True
        except Exception as e2:
            server_error = e2
            logger.error(f"Failed to start server on alternate port {alternate_port}: {str(e2)}")
            logging_utils.log_exception(logger, e2, {"port": alternate_port})
    except Exception as e:
        server_error = e
        logger.error(f"Unexpected error starting server: {str(e)}")
        logging_utils.log_exception(logger, e, {"port": args.port})
    
    if not success:
        logger.error("All server start attempts failed")
        logger.info("Try running the direct_ollama_server.py script for a simplified test")
        # Log the full traceback
        if 'server_error' in locals():
            logger.error(f"Server error details: {traceback.format_exc()}")
        
        # Exit with error code
        sys.exit(1)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        # Log shutdown
        shutdown_time = time.time()
        uptime = shutdown_time - start_time
        logger.info(f"Shutting down after {uptime:.2f} seconds due to keyboard interrupt")
        sys.exit(0)
    except Exception as e:
        # Log unexpected errors
        logger.critical(f"Fatal error: {e}")
        logging_utils.log_exception(logger, e, {"phase": "startup"})
        sys.exit(1)