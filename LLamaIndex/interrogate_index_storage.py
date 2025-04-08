# interrogate_storage.py
import json
import os
import sys

# --- Configuration ---
STORAGE_DIR = "index_storage"
FILENAMES = [
    "default_vector_store.json",
    "docstore.json",
    "graph_store.json",
    "image_vector_store.json",
    "index_store.json",
]
# --- End Configuration ---

# --- Helper Functions ---
def get_file_size(filepath):
    """Returns human-readable file size."""
    try:
        size_bytes = os.path.getsize(filepath)
        if size_bytes < 1024:
            return f"{size_bytes} B"
        elif size_bytes < 1024**2:
            return f"{size_bytes/1024:.2f} KB"
        elif size_bytes < 1024**3:
            return f"{size_bytes/1024**2:.2f} MB"
        else:
            return f"{size_bytes/1024**3:.2f} GB"
    except OSError:
        return "N/A"

def print_dict_summary(data, name="Data"):
    """Prints a summary of a dictionary."""
    num_keys = len(data)
    print(f"  - Type: Dictionary")
    print(f"  - Number of top-level keys: {num_keys}")
    if num_keys > 0:
        keys = list(data.keys())
        print(f"  - First 5 keys: {keys[:5]}")
        # Optionally print a sample value if needed
        # first_key = keys[0]
        # print(f"  - Sample value for key '{first_key}': {str(data[first_key])[:100]}...")
    else:
        print("  - Dictionary is empty.")

def print_list_summary(data, name="Data"):
    """Prints a summary of a list."""
    num_items = len(data)
    print(f"  - Type: List")
    print(f"  - Number of items: {num_items}")
    if num_items > 0:
        # Optionally print a sample item if needed
        # print(f"  - Sample first item: {str(data[0])[:100]}...")
        pass # Avoid printing potentially large items by default
    else:
        print("  - List is empty.")

def inspect_json_file(filename):
    """Loads and inspects a single JSON file."""
    filepath = os.path.join(STORAGE_DIR, filename)
    print("-" * 40)
    print(f"Inspecting: {filepath}")
    print("-" * 40)

    if not os.path.exists(filepath):
        print("  - Status: File not found.")
        return

    file_size = get_file_size(filepath)
    print(f"  - File Size: {file_size}")

    if os.path.getsize(filepath) == 0:
        print("  - Status: File is empty (0 bytes).")
        return

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print("  - Status: Successfully loaded JSON.")

        # Summarize content based on type
        if isinstance(data, dict):
            print_dict_summary(data, name=filename)
        elif isinstance(data, list):
            print_list_summary(data, name=filename)
        else:
            print(f"  - Type: {type(data).__name__}")
            print(f"  - Content (preview): {str(data)[:200]}...") # Basic preview for other types

    except json.JSONDecodeError as e:
        print(f"  - Status: Error decoding JSON.")
        print(f"  - Error Message: {e}")
        # Attempt to read first few lines for context if decode fails
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                print("  - First few lines (raw):")
                for i, line in enumerate(f):
                    if i >= 5: break
                    print(f"    {line.rstrip()}")
        except Exception as read_err:
            print(f"  - Could not read raw lines: {read_err}")

    except Exception as e:
        print(f"  - Status: An unexpected error occurred.")
        print(f"  - Error Message: {e}")

    print("\n")


# --- Main Execution ---
def main():
    """Main function to inspect all configured JSON files."""
    if not os.path.isdir(STORAGE_DIR):
        print(f"Error: Storage directory '{STORAGE_DIR}' not found.")
        print(f"Please ensure this script is run from the parent directory of '{STORAGE_DIR}'.")
        sys.exit(1)

    print(f"Starting inspection of JSON files in '{STORAGE_DIR}'...\n")
    for filename in FILENAMES:
        inspect_json_file(filename)
    print("Inspection complete.")

if __name__ == "__main__":
    main()