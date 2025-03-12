import os
import sys
import time
import logging
import traceback
import json
import uuid
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler
from functools import wraps
from datetime import datetime
from typing import Dict, Any, Optional, Callable, Union, List, Tuple

# Default log directory
LOG_DIR = os.path.join(os.getcwd(), "logs")

# Ensure log directory exists
os.makedirs(LOG_DIR, exist_ok=True)

# Log files
MAIN_LOG = os.path.join(LOG_DIR, "main.log")
ERROR_LOG = os.path.join(LOG_DIR, "error.log")
API_LOG = os.path.join(LOG_DIR, "api_calls.log")
MEMORY_LOG = os.path.join(LOG_DIR, "memory_ops.log")

# Log levels
LOG_LEVELS = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "error": logging.ERROR,
    "critical": logging.CRITICAL
}

# Request ID for tracking API calls
CURRENT_REQUEST_ID = None

class RequestIdFilter(logging.Filter):
    """Filter that adds request_id to log records."""
    def filter(self, record):
        record.request_id = CURRENT_REQUEST_ID or "-"
        return True

class ContextAdapter(logging.LoggerAdapter):
    """Adapter that allows adding context to log messages."""
    def process(self, msg, kwargs):
        context = kwargs.pop('context', {})
        if context:
            context_str = ' '.join(f'{k}={v}' for k, v in context.items())
            msg = f"{msg} [{context_str}]"
        return msg, kwargs

def get_request_id() -> str:
    """Get the current request ID or generate a new one."""
    global CURRENT_REQUEST_ID
    if CURRENT_REQUEST_ID is None:
        CURRENT_REQUEST_ID = str(uuid.uuid4())
    return CURRENT_REQUEST_ID

def set_request_id(request_id: Optional[str] = None) -> str:
    """Set the current request ID."""
    global CURRENT_REQUEST_ID
    if request_id:
        CURRENT_REQUEST_ID = request_id
    else:
        CURRENT_REQUEST_ID = str(uuid.uuid4())
    return CURRENT_REQUEST_ID

def reset_request_id() -> None:
    """Reset the current request ID."""
    global CURRENT_REQUEST_ID
    CURRENT_REQUEST_ID = None

def configure_logging(level: str = "info", 
                      console: bool = True, 
                      file: bool = True,
                      max_file_size: int = 10 * 1024 * 1024,  # 10 MB
                      backup_count: int = 10) -> None:
    """
    Configure the logging system.
    
    Args:
        level: Log level (debug, info, warning, error, critical)
        console: Whether to log to console
        file: Whether to log to files
        max_file_size: Maximum log file size in bytes before rotation
        backup_count: Number of backup files to keep
    """
    log_level = LOG_LEVELS.get(level.lower(), logging.INFO)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # Clear existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s.%(msecs)03d | %(levelname)-8s | %(name)-25s | %(request_id)s | %(message)s',
        '%Y-%m-%d %H:%M:%S'
    )
    
    # Add request ID filter to root logger
    root_logger.addFilter(RequestIdFilter())
    
    # Console handler
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        console_handler.setLevel(log_level)
        root_logger.addHandler(console_handler)
    
    # File handlers
    if file:
        # Main log (rotating by size)
        main_handler = RotatingFileHandler(
            MAIN_LOG, 
            maxBytes=max_file_size, 
            backupCount=backup_count
        )
        main_handler.setFormatter(formatter)
        main_handler.setLevel(log_level)
        root_logger.addHandler(main_handler)
        
        # Error log (keeps only ERROR and above)
        error_handler = RotatingFileHandler(
            ERROR_LOG, 
            maxBytes=max_file_size, 
            backupCount=backup_count
        )
        error_handler.setFormatter(formatter)
        error_handler.setLevel(logging.ERROR)
        root_logger.addHandler(error_handler)
        
        # API log
        api_handler = RotatingFileHandler(
            API_LOG, 
            maxBytes=max_file_size, 
            backupCount=backup_count
        )
        api_handler.setFormatter(formatter)
        api_handler.setLevel(log_level)
        api_logger = logging.getLogger('api')
        api_logger.propagate = False  # Don't propagate to root
        api_logger.addHandler(api_handler)
        
        # Memory log
        memory_handler = RotatingFileHandler(
            MEMORY_LOG, 
            maxBytes=max_file_size, 
            backupCount=backup_count
        )
        memory_handler.setFormatter(formatter)
        memory_handler.setLevel(log_level)
        memory_logger = logging.getLogger('memory')
        memory_logger.propagate = False  # Don't propagate to root
        memory_logger.addHandler(memory_handler)

def get_logger(name: str, context: Optional[Dict[str, Any]] = None) -> Union[logging.Logger, ContextAdapter]:
    """
    Get a logger with the given name and optional context.
    
    Args:
        name: Logger name
        context: Optional context dictionary to include in log messages
        
    Returns:
        Logger or LoggerAdapter with context
    """
    logger = logging.getLogger(name)
    
    if context:
        return ContextAdapter(logger, {})
    return logger

def _sanitize_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Sanitize a payload to remove sensitive information.
    
    Args:
        payload: The payload to sanitize
        
    Returns:
        Sanitized payload
    """
    sanitized = payload.copy()
    
    # List of keys that might contain sensitive data
    sensitive_keys = ['api_key', 'authorization', 'password', 'token', 'secret']
    
    for key in payload:
        if any(sensitive_word in key.lower() for sensitive_word in sensitive_keys):
            sanitized[key] = "***REDACTED***"
    
    return sanitized

def _format_payload(payload: Any) -> str:
    """
    Format payload for logging, limiting its size.
    
    Args:
        payload: The payload to format
        
    Returns:
        Formatted payload string
    """
    try:
        if payload is None:
            return "None"
            
        if isinstance(payload, (dict, list)):
            # Sanitize before converting to string
            if isinstance(payload, dict):
                payload = _sanitize_payload(payload)
                
            # Convert to string with nice formatting
            payload_str = json.dumps(payload, indent=2, ensure_ascii=False)
            
            # Truncate if too long
            max_length = 1000
            if len(payload_str) > max_length:
                return payload_str[:max_length] + "... [truncated]"
            return payload_str
        
        # For other types, convert to string
        return str(payload)
    except Exception as e:
        return f"[Error formatting payload: {e}]"

def log_api_call(logger: logging.Logger, 
                method: str, 
                url: str, 
                headers: Optional[Dict[str, str]] = None, 
                payload: Any = None,
                response: Any = None,
                status_code: Optional[int] = None,
                error: Optional[Exception] = None,
                duration_ms: Optional[float] = None) -> None:
    """
    Log an API call with request and response details.
    
    Args:
        logger: Logger to use
        method: HTTP method
        url: URL
        headers: Request headers
        payload: Request payload
        response: Response data
        status_code: Response status code
        error: Exception if any occurred
        duration_ms: Call duration in milliseconds
    """
    # Create a unique ID for this API call if not already set
    request_id = get_request_id()
    
    # Format headers (sanitize auth headers)
    formatted_headers = None
    if headers:
        sanitized_headers = headers.copy()
        for key in headers:
            if 'auth' in key.lower() or 'key' in key.lower() or 'token' in key.lower():
                sanitized_headers[key] = "***REDACTED***"
        formatted_headers = json.dumps(sanitized_headers)
    
    # Format request
    request_info = f"{method} {url}"
    if formatted_headers:
        request_info += f"\nHeaders: {formatted_headers}"
    if payload:
        formatted_payload = _format_payload(payload)
        request_info += f"\nPayload: {formatted_payload}"
    
    # Log request
    logger.info(f"API Call [{request_id}]: {request_info}")
    
    # Format and log response if provided
    if status_code is not None:
        response_info = f"Status: {status_code}"
        
        if duration_ms is not None:
            response_info += f", Duration: {duration_ms:.2f}ms"
            
        if response:
            formatted_response = _format_payload(response)
            response_info += f"\nResponse: {formatted_response}"
            
        if error:
            response_info += f"\nError: {str(error)}"
            
        if status_code >= 400:
            logger.error(f"API Response [{request_id}]: {response_info}")
        else:
            logger.info(f"API Response [{request_id}]: {response_info}")

def log_memory_operation(logger: logging.Logger,
                        operation: str,
                        user_id: str,
                        success: bool,
                        details: Optional[Dict[str, Any]] = None,
                        error: Optional[Exception] = None,
                        duration_ms: Optional[float] = None) -> None:
    """
    Log a memory operation.
    
    Args:
        logger: Logger to use
        operation: Operation type (add, search, etc.)
        user_id: User ID
        success: Whether the operation was successful
        details: Operation details
        error: Exception if any occurred
        duration_ms: Operation duration in milliseconds
    """
    status = "SUCCESS" if success else "FAILED"
    
    message = f"Memory {operation.upper()} [{status}] User: {user_id}"
    
    if duration_ms is not None:
        message += f", Duration: {duration_ms:.2f}ms"
        
    if details:
        formatted_details = _format_payload(details)
        message += f"\nDetails: {formatted_details}"
        
    if error:
        message += f"\nError: {str(error)}"
        
    if success:
        logger.info(message)
    else:
        logger.error(message)

def log_exception(logger: logging.Logger, 
                 exc: Exception, 
                 context: Optional[Dict[str, Any]] = None) -> None:
    """
    Log an exception with full stack trace and context.
    
    Args:
        logger: Logger to use
        exc: Exception
        context: Additional context
    """
    exc_type, exc_value, exc_traceback = sys.exc_info()
    stack_trace = ''.join(traceback.format_exception(exc_type, exc_value, exc_traceback))
    
    message = f"Exception: {exc.__class__.__name__}: {str(exc)}"
    
    if context:
        formatted_context = _format_payload(context)
        message += f"\nContext: {formatted_context}"
        
    message += f"\nStack trace:\n{stack_trace}"
    
    logger.error(message)

def timed_operation(logger: Optional[logging.Logger] = None, operation: Optional[str] = None):
    """
    Decorator to time an operation and log its duration.
    
    Args:
        logger: Logger to use (if None, uses operation name as logger name)
        operation: Operation name (if None, uses function name)
        
    Returns:
        Decorated function
    """
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            nonlocal logger, operation
            
            # Get logger if not provided
            if logger is None:
                logger = logging.getLogger(func.__module__)
                
            # Get operation name if not provided
            if operation is None:
                operation = func.__name__
                
            # Start timer
            start_time = time.time()
            
            # Generate request ID if not already set
            request_id = get_request_id()
            
            logger.debug(f"Starting {operation} [request_id={request_id}]")
            
            try:
                # Call the function
                result = func(*args, **kwargs)
                
                # Calculate duration
                duration_ms = (time.time() - start_time) * 1000
                
                # Log success
                logger.info(f"Completed {operation} in {duration_ms:.2f}ms [request_id={request_id}]")
                
                return result
            except Exception as e:
                # Calculate duration
                duration_ms = (time.time() - start_time) * 1000
                
                # Log error
                logger.error(f"Failed {operation} after {duration_ms:.2f}ms: {str(e)} [request_id={request_id}]")
                
                # Re-raise
                raise
                
        return wrapper
    return decorator

def analyze_logs(days: int = 1) -> Dict[str, Any]:
    """
    Analyze logs to generate a summary of errors, warnings, and metrics.
    
    Args:
        days: Number of days to analyze
        
    Returns:
        Dictionary with analysis results
    """
    # This is a placeholder for a more comprehensive log analysis function
    # In a real implementation, we would parse the log files and generate metrics
    
    now = datetime.now()
    cutoff = now.timestamp() - days * 24 * 60 * 60
    
    result = {
        "period": f"Last {days} days",
        "generated_at": now.isoformat(),
        "error_count": 0,
        "warning_count": 0,
        "api_calls": {
            "total": 0,
            "success": 0,
            "failure": 0,
            "avg_duration_ms": 0
        },
        "memory_operations": {
            "total": 0,
            "success": 0,
            "failure": 0,
            "avg_duration_ms": 0
        },
        "top_errors": [],
        "top_endpoints": []
    }
    
    return result

# Initialize logging with defaults
configure_logging(level="info", console=True, file=True)