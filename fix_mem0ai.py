"""
Script to diagnose and fix mem0ai import issues
"""

import os
import sys
import site
import importlib
import subprocess
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_package_path():
    """Use pip to get the mem0ai package path"""
    try:
        result = subprocess.run(
            ["pip", "show", "mem0ai"], 
            capture_output=True, 
            text=True, 
            check=True
        )
        
        # Parse the output to find the location
        for line in result.stdout.split('\n'):
            if line.startswith('Location:'):
                return line.split(':', 1)[1].strip()
        
        return None
    except subprocess.CalledProcessError as e:
        logger.error(f"Error running pip show: {e}")
        logger.error(f"Stderr: {e.stderr}")
        return None

def fix_import_issues():
    """Try to fix mem0ai import issues"""
    logger.info(f"Python executable: {sys.executable}")
    logger.info(f"Python version: {sys.version}")
    
    # Get user site-packages directory
    user_site = site.getusersitepackages()
    logger.info(f"User site-packages: {user_site}")
    
    # Get system site-packages directories
    site_packages = site.getsitepackages()
    logger.info(f"System site-packages: {site_packages}")
    
    # Current sys.path
    logger.info(f"Current sys.path: {sys.path}")
    
    # Get the package path from pip
    package_path = get_package_path()
    logger.info(f"Package path from pip: {package_path}")
    
    if package_path:
        if package_path in sys.path:
            logger.info(f"Package path already in sys.path")
        else:
            logger.info(f"Adding package path to sys.path")
            sys.path.append(package_path)
    
    # Try to import mem0ai after path modifications
    try:
        logger.info("Trying to import mem0ai module...")
        
        # Force a reload of sys.path
        importlib.invalidate_caches()
        
        import mem0ai
        from mem0ai import Memory
        
        logger.info(f"Successfully imported mem0ai (version: {getattr(mem0ai, '__version__', 'unknown')})")
        logger.info(f"Module location: {mem0ai.__file__}")
        
        # Try creating a Memory instance
        config = {
            "llm": {
                "provider": "ollama",
                "config": {
                    "model": "llama3",
                    "base_url": "http://localhost:11434"
                }
            },
            "embedder": {
                "provider": "ollama", 
                "config": {
                    "model": "nomic-embed-text",
                    "base_url": "http://localhost:11434"
                }
            },
            "vector_db": {
                "provider": "in_memory"
            }
        }
        
        memory = Memory.from_config(config)
        logger.info("Successfully created Memory instance")
        
        # Create a simple test file to verify imports
        with open('verify_mem0ai_import.py', 'w') as f:
            f.write("""
import sys
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

logger.info(f"Python executable: {sys.executable}")
logger.info(f"Python path: {sys.path}")

try:
    import mem0ai
    from mem0ai import Memory
    logger.info(f"Successfully imported mem0ai from {mem0ai.__file__}")
except ImportError as e:
    logger.error(f"Failed to import mem0ai: {e}")
""")
        
        logger.info("Created verification script at verify_mem0ai_import.py")
        logger.info("To fix the main application, try adding this near the top:")
        logger.info(f"""
import sys
sys.path.append("{package_path}")
import importlib
importlib.invalidate_caches()
""")
        
        return True
        
    except ImportError as e:
        logger.error(f"Failed to import mem0ai: {e}")
        return False
    except Exception as e:
        logger.error(f"Error during import: {e}")
        return False

if __name__ == "__main__":
    fix_import_issues()
