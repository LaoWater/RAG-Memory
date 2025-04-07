# First Contact with low-level memory management: a success.
# Escaping the overwhelming 3rd party apps and libraries until there is full understanding and grasping, up to the very matrix multiplications and storing

# Got it to work in First Contact_Gemini with full context threshold based on new prompt (Groq - wow such fast inference!)
# (aka Should i Retrieve any full memories of this conversation to properly answer in context?) -
# If prompt hits an activation threshold regarding a full memory - retrieve - not doing anything with summaries.
# Shows clear grasp of context and flow.



import os
import uuid
import numpy as np
from typing import Dict, List
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
from basic_inference_fast import groq_inference

# Set up local embedding model
EMBEDDING_MODEL = SentenceTransformer('all-MiniLM-L6-v2')


def embed_text(text: str) -> np.ndarray:
    """Generate embeddings using local model"""
    return EMBEDDING_MODEL.encode(text, convert_to_numpy=True)


class AgenticMemory:
    def __init__(self):
        self.vector_db: Dict[str, np.ndarray] = {}  # UUID to embedding
        self.full_text_db: Dict[str, str] = {}  # UUID to text
        print("Memory initialized - vector_db and full_text_db created")

    def store_full_text(self, prompt: str, response: str) -> str:
        """Store conversation with generated UUID"""
        text = prompt + response
        print(f"\nStoring text (length: {len(text)}): {text[:50]}...")
        embedding = embed_text(text)
        entry_id = str(uuid.uuid4())

        self.vector_db[entry_id] = embedding
        self.full_text_db[entry_id] = text
        print(f"Stored with ID: {entry_id}")
        print(f"Current vector_db size: {len(self.vector_db)}")
        print(f"Current full_text_db size: {len(self.full_text_db)}")
        return entry_id

    def summarize_pair(self, prompt: str, response: str) -> str:
        """Generate and store summary with its own entry"""
        print("\nCreating summary...")
        summary_prompt = f"Summarize the following conversation: {prompt} {response}"
        summary_response = groq_inference(model='meta-llama/llama-4-scout-17b-16e-instruct', prompt=summary_prompt)
        summary_text = summary_response
        print(f"Summary created: {summary_text[:50]}...")

        return self.store_full_text(summary_prompt, summary_text)

    def query_vector_db(self, user_prompt: str) -> List[str]:
        """Find similar entries using cosine similarity"""
        print(f"\nQuerying vector DB with: '{user_prompt}'")
        user_embedding = embed_text(user_prompt)
        user_norm = np.linalg.norm(user_embedding)
        similar_ids = []
        print(f"User embedding shape: {user_embedding.shape}")
        print(f"User norm: {user_norm}")

        for entry_id, embedding in self.vector_db.items():
            dot_product = np.dot(user_embedding, embedding)
            embedding_norm = np.linalg.norm(embedding)

            if embedding_norm == 0 or user_norm == 0:
                continue

            similarity = dot_product / (user_norm * embedding_norm)
            print(f"Comparison with {entry_id[:8]}...: similarity={similarity:.4f}")
            if similarity > 0.4:
                similar_ids.append(entry_id)
                print(f"Above threshold! Added {entry_id[:8]}...")

        print(f"Found {len(similar_ids)} similar entries")
        return similar_ids

    def retrieve_full_text(self, entry_ids: List[str]) -> List[str]:
        """Retrieve texts by their UUIDs"""
        print(f"\nRetrieving full text for {len(entry_ids)} entries")
        results = []
        for uid in entry_ids:
            if uid in self.full_text_db:
                print(f"Found entry {uid[:8]}...: {self.full_text_db[uid][:50]}...")
                results.append(self.full_text_db[uid])
            else:
                print(f"Entry {uid[:8]}... not found in full_text_db")
        return results


# Initialize Agentic Memory
memory = AgenticMemory()

# First response - Stateless Answer
prompt = "Hello. Poem about the meaning of life in 2 verses."
response = groq_inference(model='meta-llama/llama-4-scout-17b-16e-instruct', prompt=prompt)

print(response + '\n')

# Store conversation and summary
conversation_id = memory.store_full_text(prompt, response)
summary_id = memory.summarize_pair(prompt, response)

# Second Inference
user_prompt = "Continue Poem."
similar_entries = memory.query_vector_db(user_prompt)
contexts = memory.retrieve_full_text(similar_entries)
print(f'Context: {contexts} \n')


# Generate response with context
augmented_prompt = user_prompt + "\nContext:\n" + "\n".join(contexts)
second_response = groq_inference(model='meta-llama/llama-4-scout-17b-16e-instruct', prompt=augmented_prompt)

print(second_response)

# Store new interaction
memory.store_full_text(user_prompt, second_response)
memory.summarize_pair(user_prompt, second_response)

