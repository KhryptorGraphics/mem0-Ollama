"""
Health check utilities for mem0-ollama services.

This module provides functions to check the health and availability of
dependent services like Ollama and mem0 vector database.
"""

import requests
import time
import logging
from typing import Dict, Any, Tuple, List, Optional
from config import OLLAMA_HOST, MEM0_HOST, DEFAULT_MODEL

# Set up logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ServiceStatus:
    """Status of external service dependencies"""
    def __init__(self, name: str, available: bool, details: Dict[str, Any] = None):
        self.name = name
        self.available = available
        self.details = details or {}
        self.last_checked = time.time()
    
    def __str__(self) -> str:
        status = "Available" if self.available else "Unavailable"
        return f"{self.name}: {status}"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert status to dictionary for API responses"""
        return {
            "name": self.name,
            "available": self.available,
            "last_checked": self.last_checked,
            "details": self.details
        }

def check_ollama_health(host: Optional[str] = None, timeout: int = 5) -> ServiceStatus:
    """
    Check if Ollama service is available and working.
    
    Args:
        host: Ollama host URL (defaults to config value)
        timeout: Request timeout in seconds
        
    Returns:
        ServiceStatus object with availability info
    """
    if not host:
        # Get from config, but force localhost if using 0.0.0.0
        host = OLLAMA_HOST
        if '0.0.0.0' in host:
            host = 'http://localhost:11434'
    
    # Make sure host has proper URL scheme
    if not host.startswith(('http://', 'https://')):
        host = f"http://{host}"
    
    try:
        # Try to call the Ollama API tags endpoint
        url = f"{host}/api/tags"
        response = requests.get(url, timeout=timeout)
        
        if response.status_code == 200:
            # Get available models from response
            data = response.json()
            models = [model.get('name') for model in data.get('models', [])]
            details = {
                "models_available": len(models),
                "models": models,
                "api_version": data.get('api_version', 'unknown')
            }
            
            # Check if configured model is available
            default_model_available = DEFAULT_MODEL in models
            
            return ServiceStatus(
                name="Ollama",
                available=True,
                details={
                    **details,
                    "default_model_available": default_model_available,
                    "host": host
                }
            )
        else:
            return ServiceStatus(
                name="Ollama",
                available=False,
                details={
                    "host": host,
                    "status_code": response.status_code,
                    "error": "Received non-200 status code"
                }
            )
    except requests.exceptions.Timeout:
        return ServiceStatus(
            name="Ollama",
            available=False,
            details={
                "host": host,
                "error": "Connection timeout"
            }
        )
    except requests.exceptions.ConnectionError:
        return ServiceStatus(
            name="Ollama",
            available=False,
            details={
                "host": host,
                "error": "Connection refused"
            }
        )
    except Exception as e:
        return ServiceStatus(
            name="Ollama",
            available=False,
            details={
                "host": host,
                "error": str(e)
            }
        )

def check_qdrant_health(host: Optional[str] = None, timeout: int = 5) -> ServiceStatus:
    """
    Check if Qdrant vector database is available and working.
    
    Args:
        host: Qdrant host URL (defaults to config value)
        timeout: Request timeout in seconds
        
    Returns:
        ServiceStatus object with availability info
    """
    host = host or MEM0_HOST
    
    # Make sure host has proper URL scheme
    if not host.startswith(('http://', 'https://')):
        host = f"http://{host}"
    
    try:
        # Check Qdrant collections endpoint which is specific to Qdrant
        collections_url = f"{host}/collections"
        response = requests.get(collections_url, timeout=timeout)
        
        if response.status_code == 200:
            # Get collection information
            collections_data = response.json()
            collections = collections_data.get("collections", [])
            collection_names = [c.get("name") for c in collections]
            
            # Check for mem0_memories collection
            mem0_collection_exists = "mem0_memories" in collection_names
            
            return ServiceStatus(
                name="Qdrant Vector DB",
                available=True,
                details={
                    "host": host,
                    "collections_count": len(collections),
                    "collections": collection_names,
                    "mem0_collection_exists": mem0_collection_exists
                }
            )
        else:
            # Try a basic connection test to see if the server is up
            try:
                basic_response = requests.get(host, timeout=timeout)
                basic_connected = basic_response.status_code < 500
                
                return ServiceStatus(
                    name="Qdrant Vector DB",
                    available=basic_connected,
                    details={
                        "host": host,
                        "status_code": basic_response.status_code,
                        "collections_error": f"Status code: {response.status_code}",
                        "note": "Server is reachable but collections endpoint failed"
                    }
                )
            except Exception:
                return ServiceStatus(
                    name="Qdrant Vector DB",
                    available=False,
                    details={
                        "host": host,
                        "status_code": response.status_code,
                        "error": "Failed to connect to collections endpoint"
                    }
                )
    except requests.exceptions.Timeout:
        return ServiceStatus(
            name="Qdrant Vector DB",
            available=False,
            details={
                "host": host,
                "error": "Connection timeout"
            }
        )
    except requests.exceptions.ConnectionError:
        return ServiceStatus(
            name="Qdrant Vector DB",
            available=False,
            details={
                "host": host,
                "error": "Connection refused"
            }
        )
    except Exception as e:
        return ServiceStatus(
            name="Qdrant Vector DB",
            available=False,
            details={
                "host": host,
                "error": str(e)
            }
        )

def check_mem0_health() -> ServiceStatus:
    """
    Check if mem0ai library is available and properly configured.
    
    Returns:
        ServiceStatus object with availability info
    """
    try:
        # Import mem0ai integration to check its status
        from final_mem0_adapter import HAS_MEM0AI
        
        # Use the flag from the adapter
        if HAS_MEM0AI:
            return ServiceStatus(
                name="mem0ai Library",
                available=True,
                details={
                    "import_status": "Successfully imported"
                }
            )
        else:
            return ServiceStatus(
                name="mem0ai Library",
                available=False,
                details={
                    "import_status": "Available but failed to import correctly",
                    "note": "Using fallback memory implementation"
                }
            )
    except ImportError:
        return ServiceStatus(
            name="mem0ai Library",
            available=False,
            details={
                "import_status": "Not installed or not in path",
                "error": "Module not found"
            }
        )
    except Exception as e:
        return ServiceStatus(
            name="mem0ai Library",
            available=False,
            details={
                "import_status": "Error during import",
                "error": str(e)
            }
        )

def check_all_services() -> Dict[str, ServiceStatus]:
    """
    Check health of all required services.
    
    Returns:
        Dictionary of service statuses
    """
    ollama_status = check_ollama_health()
    mem0_status = check_mem0_health()
    qdrant_status = check_qdrant_health()
    
    # All available is true only if all services are available
    all_available = ollama_status.available and mem0_status.available and qdrant_status.available
    
    return {
        "ollama": ollama_status,
        "mem0": mem0_status,
        "qdrant": qdrant_status,
        "all_available": all_available
    }

def get_system_health() -> Dict[str, Any]:
    """
    Get comprehensive system health information.
    
    Returns:
        Dictionary with system health details
    """
    services = check_all_services()
    
    # Check if we're using mem0ai with Qdrant
    vector_db_info = {}
    try:
        from final_mem0_adapter import Mem0Adapter
        adapter = Mem0Adapter()
        memory_system_info = adapter.get_memory_system_info()
        using_mem0ai = memory_system_info.get("current_mode") == "mem0ai"
        vector_db_info = {
            "memory_system": memory_system_info
        }
    except Exception as e:
        using_mem0ai = False
        vector_db_info = {
            "memory_system_error": str(e)
        }
    
    return {
        "services": {
            name: status.to_dict() for name, status in services.items() 
            if name != "all_available"
        },
        "all_services_available": services["all_available"],
        "timestamp": time.time(),
        "vector_db_info": vector_db_info
    }

if __name__ == '__main__':
    # When run directly, print health check information
    import json
    
    health = get_system_health()
    print(json.dumps(health, indent=2))
    
    # Print status summary
    services = check_all_services()
    for name, status in services.items():
        if name != "all_available":
            print(status)
    
    overall = "All services available" if services["all_available"] else "Some services unavailable"
    print(f"Overall status: {overall}")
