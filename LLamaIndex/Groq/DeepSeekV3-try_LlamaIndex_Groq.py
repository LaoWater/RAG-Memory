"""
- Using Llama_index specific Groq high-level library.
Things have 10x simplified
"""

import os
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.llms.groq import Groq as GroqLLM


def main():
    print("Setting up Groq LLM in llama_index settings...")
    # Configure LlamaIndex settings
    Settings.llm = GroqLLM(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        api_key=os.environ.get("GROQ_API_KEY"),
        temperature=0.1,
        max_tokens=1024
    )

    print("Reading, Indexing and Embedding document...")
    # Load data from data.txt in current directory
    documents = SimpleDirectoryReader(input_files=["data.txt"]).load_data()

    # Create index
    index = VectorStoreIndex.from_documents(documents)

    # Create query engine
    query_engine = index.as_query_engine()

    # Perform query using document context
    query = "Based on the document, write a one-verse poem about Coconuts and AI."
    print("Generating Inference..")
    response = query_engine.query(query)

    print("Context-based Response:")
    print(response.response)

    # # Direct inference example without context
    # direct_response = groq_llm.generate("Briefly explain quantum entanglement in relation to Coconuts.")
    # print("\nDirect Inference Response:")
    # print(direct_response)


if __name__ == "__main__":
    main()

