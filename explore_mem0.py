"""
Script to explore the mem0 repository structure and find examples
"""

import os
import sys

def print_directory_structure(start_path, indent=0, max_depth=3, current_depth=0, file_extensions=None):
    """
    Print the directory structure with indentation
    """
    if current_depth > max_depth:
        print(f"{' ' * indent}[...]")
        return
        
    if os.path.isdir(start_path):
        print(f"{' ' * indent}[{os.path.basename(start_path)}]")
        items = os.listdir(start_path)
        
        # Sort directories first, then files
        dirs = [i for i in items if os.path.isdir(os.path.join(start_path, i))]
        files = [i for i in items if os.path.isfile(os.path.join(start_path, i))]
        
        # Sort alphabetically within each group
        dirs.sort()
        files.sort()
        
        # Process directories first
        for item in dirs:
            if item.startswith('.'):
                continue  # Skip hidden directories
            print_directory_structure(os.path.join(start_path, item), indent + 2, 
                                    max_depth, current_depth + 1, file_extensions)
        
        # Then process files
        for item in files:
            if file_extensions is None or any(item.endswith(ext) for ext in file_extensions):
                print(f"{' ' * (indent + 2)}{item}")
    else:
        print(f"{' ' * indent}{os.path.basename(start_path)}")

def find_files_with_content(start_path, search_term, file_extensions=None):
    """
    Find files containing the search term
    """
    found_files = []
    
    for root, dirs, files in os.walk(start_path):
        for file in files:
            if file_extensions is None or any(file.endswith(ext) for ext in file_extensions):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        if search_term.lower() in content.lower():
                            found_files.append(file_path)
                except Exception as e:
                    # Skip files that can't be read as text
                    pass
    
    return found_files

if __name__ == "__main__":
    repo_path = os.path.join(os.getcwd(), "external", "mem0")
    
    # Print the directory structure with focus on Python files
    print("\n=== DIRECTORY STRUCTURE (FOCUSING ON PYTHON FILES) ===")
    print_directory_structure(repo_path, file_extensions=['.py'])
    
    # Find examples in the repository
    print("\n=== SEARCHING FOR OLLAMA EXAMPLES ===")
    ollama_files = find_files_with_content(repo_path, "ollama", file_extensions=['.py'])
    for file in ollama_files:
        relative_path = os.path.relpath(file, repo_path)
        print(f"Found Ollama reference in: {relative_path}")
    
    # Look for memory implementation files
    print("\n=== MAIN MEMORY IMPLEMENTATION FILES ===")
    memory_files = find_files_with_content(repo_path, "class Memory", file_extensions=['.py'])
    for file in memory_files:
        relative_path = os.path.relpath(file, repo_path)
        print(f"Memory class implementation: {relative_path}")
    
    # Look specifically for example files
    print("\n=== EXAMPLE FILES ===")
    example_files = []
    for root, dirs, files in os.walk(repo_path):
        if "example" in root.lower() or "demo" in root.lower():
            for file in files:
                if file.endswith('.py'):
                    example_files.append(os.path.join(root, file))
    
    for file in example_files:
        relative_path = os.path.relpath(file, repo_path)
        print(f"Example file: {relative_path}")
