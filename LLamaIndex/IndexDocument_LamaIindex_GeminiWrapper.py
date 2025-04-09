"""
LlamaIndex integration with Gemini, using llamaindex gemini library (high-level setup)

Here we also simulate production-like deployment, where the embedded document indexing is saved for ease of access at inference time

"""


import os
import logging # Optional: for more detailed logs
import sys   # Optional: for detailed logs

from llama_index.core import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    Settings,
    StorageContext,     # Needed for persistence
    load_index_from_storage # Needed for loading
)
from llama_index.llms.gemini import Gemini

# Optional: Setup logging for more insight
# logging.basicConfig(stream=sys.stdout, level=logging.INFO)
# logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

# --- Configuration ---
PERSIST_DIR = "./index_storage" # Directory to save/load the index
DOC_PATH = "./data.txt"        # Path to your document

# Step 1: Setup Gemini API Key
google_api_key = os.getenv("GOOGLE_API_KEY")
if not google_api_key:
    raise ValueError(
        "GOOGLE_API_KEY environment variable not set. "
        "Please set it before running the script."
    )

# Choose your Gemini model name - REQUIRES 'models/' prefix
gemini_model_name = 'models/gemini-2.0-flash'

# Step 2: Configure LlamaIndex Settings (Needed for both building and loading)
# The LLM needs to be available when loading the index so it knows how to query.
print("Configuring LLM...")
# Note the DeprecationWarning: Consider switching to llama-index-llms-google later
Settings.llm = Gemini(model_name=gemini_model_name, api_key=google_api_key)
# Optional: Configure an embedding model if needed (must be same as used for building)
# Settings.embed_model = ...


# --- Index Loading/Building ---

index = None # Initialize index variable
if os.path.exists(PERSIST_DIR):
    # Load the existing index
    print(f"Loading index from {PERSIST_DIR}...")
    storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
    index = load_index_from_storage(storage_context)
    print("Index loaded successfully.")
else:
    # Build the index if it doesn't exist
    print(f"Index not found at {PERSIST_DIR}. Building index...")

    print(f"Loading documents from '{DOC_PATH}'...")
    documents = SimpleDirectoryReader(input_files=[DOC_PATH]).load_data()

    # Step 3-4: Build the Vector Store Index
    print("Creating index...")
    index = VectorStoreIndex.from_documents(documents)
    print("Index created successfully.")

    # Step 5: Persist the index to disk
    print(f"Persisting index to {PERSIST_DIR}...")
    # Ensure the directory exists before persisting
    os.makedirs(PERSIST_DIR, exist_ok=True)
    index.storage_context.persist(persist_dir=PERSIST_DIR)
    print("Index persisted successfully.")

# --- Querying (Now uses the loaded or newly built index) ---

if index is None:
     raise RuntimeError("Index could not be loaded or built.")

# Step 6: Ask it questions
print("Creating query engine...")
query_engine = index.as_query_engine()

query = "Why are coconuts important in AI development?"
print(f"\nQuerying index: '{query}'")
response = query_engine.query(query)

print("\nResponse:")
print(response)



# Example of another query
query2 = "Summarize the document."
print(f"\nQuerying index: '{query2}'")
response2 = query_engine.query(query2)
print("\nResponse:")
print(response2)