import os
import logging
import sys

from llama_index.core import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    Settings,
    StorageContext,
    load_index_from_storage,
)
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.retrievers import VectorIndexRetriever, BaseRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.schema import NodeWithScore
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.postprocessor import SentenceTransformerRerank
from typing import List

# Logging
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

# --- Config ---
PERSIST_DIR = "./index_storage"
DOC_PATH = "./data.txt"
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY environment variable not set.")

Settings.llm = GoogleGenAI(model="gemini-2.0-flash")

# --- Load or Build Index ---
index = None
documents = None
nodes = None

if os.path.exists(PERSIST_DIR):
    print(f"Loading index from {PERSIST_DIR}...")
    storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
    index = load_index_from_storage(storage_context)
    print("Index loaded successfully.")
    documents = SimpleDirectoryReader(input_files=[DOC_PATH]).load_data()
else:
    print("Building index from documents...")
    documents = SimpleDirectoryReader(input_files=[DOC_PATH]).load_data()
    index = VectorStoreIndex.from_documents(documents)
    os.makedirs(PERSIST_DIR, exist_ok=True)
    index.storage_context.persist(persist_dir=PERSIST_DIR)
    print("Index built and persisted.")

# --- Parse documents into Nodes for BM25 ---
print("Parsing documents into nodes for BM25...")
splitter = SentenceSplitter(chunk_size=256)
nodes = splitter.get_nodes_from_documents(documents)

# --- Set up BM25, Vector, and Hybrid Retriever ---
print("Setting up retrievers...")
bm25_retriever = BM25Retriever.from_defaults(nodes=nodes, similarity_top_k=min(5, len(nodes)))
vector_retriever = VectorIndexRetriever(index=index, similarity_top_k=min(5, len(nodes)))


class HybridRetriever(BaseRetriever):
    def __init__(self, bm25_retriever, vector_retriever):
        super().__init__()
        self.bm25_retriever = bm25_retriever
        self.vector_retriever = vector_retriever

    def _retrieve(self, query: str) -> List[NodeWithScore]:
        bm25_nodes = self.bm25_retriever.retrieve(query)
        vector_nodes = self.vector_retriever.retrieve(query)
        combined = {n.node.node_id: n for n in bm25_nodes}
        for node in vector_nodes:
            combined.setdefault(node.node.node_id, node)
        return list(combined.values())


hybrid_retriever = HybridRetriever(bm25_retriever, vector_retriever)

# --- Setup reranker ---
print("Setting up reranker...")
reranker = SentenceTransformerRerank(
    model="cross-encoder/ms-marco-MiniLM-L-2-v2",
    top_n=5
)

# --- Query Engine ---
print("Building query engine...")
query_engine = RetrieverQueryEngine.from_args(
    retriever=hybrid_retriever,
    node_postprocessors=[reranker]
)


# --- Debug helpers ---
def debug_retrieval(query: str):
    print("\n🔍 BM25 RESULTS:")
    for i, n in enumerate(bm25_retriever.retrieve(query)):
        print(f"  {i + 1}. {n.get_text()[:100]}...")

    print("\n🔎 VECTOR RESULTS:")
    for i, n in enumerate(vector_retriever.retrieve(query)):
        print(f"  {i + 1}. {n.get_text()[:100]}...")

    print("\n🧪 HYBRID RESULTS (PRE-RERANK):")
    for i, n in enumerate(hybrid_retriever.retrieve(query)):
        print(f"  {i + 1}. {n.get_text()[:100]}...")


# --- Query examples ---
query_1 = "Why are coconuts important in AI development?"
print(f"\n\n🤖 QUERY 1: {query_1}")
debug_retrieval(query_1)
response_1 = query_engine.query(query_1)
print("\n🧠 FINAL RESPONSE:")
print(response_1)

query_2 = "Summarize the document."
print(f"\n\n🤖 QUERY 2: {query_2}")
debug_retrieval(query_2)
response_2 = query_engine.query(query_2)
print("\n🧠 FINAL RESPONSE:")
print(response_2)
