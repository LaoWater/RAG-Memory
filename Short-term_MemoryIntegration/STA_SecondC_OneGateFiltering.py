# And Seond_contact -  and more complex Architecture - (can ponder switching to Groq cause of Inference Latency)
# more Agentic System Oriented, with relevance filtering at the memory Gate.
# First - looks at 'S' Type memories - and ponders if it should retrieve the Full Memory (type 'F') for dhamma answer.
# Then Retrieves Full Memories if needed (top k=3)  - and builds the new entailing token window for the LLM - ending with current prompt.
# Shows deeper Modelling to the Mind - thus feeling more alligned.
# All dealt in tokenized, embedded text and Matrix Multiplications. Fascinating.


import faiss
import os
import numpy as np
from sentence_transformers import SentenceTransformer
import google.generativeai as genai

# Configuration
EMBEDDING_MODEL = SentenceTransformer('all-mpnet-base-v2')
EMBEDDING_SIZE = 768
FAISS_INDEX_S = faiss.IndexFlatL2(EMBEDDING_SIZE)
FAISS_INDEX_F = faiss.IndexFlatL2(EMBEDDING_SIZE)

# Configure Gemini
gemini_api_key = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=gemini_api_key)
gemini_model = genai.GenerativeModel('gemini-1.5-flash')


class ConversationMemory:
    def __init__(self):
        self.memories = []
        self.current_id = 0

    class MemoryItem:
        def __init__(self, memory_id, content, embedding, memory_type):
            self.id = memory_id
            self.content = content
            self.embedding = embedding
            self.type = memory_type  # 'S' or 'F'

    def store_memory(self, text, memory_type):
        embedding = EMBEDDING_MODEL.encode(text)
        item = self.MemoryItem(
            memory_id=self.current_id,
            content=text,
            embedding=embedding,
            memory_type=memory_type
        )
        self.memories.append(item)

        # Add to FAISS index
        if memory_type == 'S':
            FAISS_INDEX_S.add(np.array([embedding]))
        else:
            FAISS_INDEX_F.add(np.array([embedding]))

        self.current_id += 1

    def retrieve_memories(self, query_embedding, memory_type, top_k=3):
        """Simple retrieval without temporal decay"""
        scores, indices = self._search_index(query_embedding, memory_type, top_k)

        results = []
        if indices.shape[0] == 0:  # No results
            return results

        for i in range(min(top_k, indices.shape[1])):
            idx = indices[0][i]
            if idx >= 0 and idx < len(self.memories):
                results.append((scores[0][i], self.memories[idx]))

        return results

    def _search_index(self, query_embedding, memory_type, k=5):
        index = FAISS_INDEX_S if memory_type == 'S' else FAISS_INDEX_F
        if index.ntotal == 0:  # Check if index is empty
            return np.empty((0, k)), np.empty((0, k), dtype=int)
        return index.search(np.array([query_embedding]), k)


class AgenticSystem:
    def __init__(self):
        self.memory = ConversationMemory()
        self.summarizer_prompt = """Please create a concise summary of this conversation pair that preserves:
- Core concepts
- Emotional tone
- Key entities
- Open questions
Keep under 3 sentences. Return only the summary."""

    def generate_response(self, user_prompt):
        # Memory Retrieval Phase
        prompt_embedding = EMBEDDING_MODEL.encode(user_prompt)

        # Retrieve summaries
        summary_memories = self.memory.retrieve_memories(prompt_embedding, 'S')

        # LLM-based relevance check
        retrieval_decision = self._should_retrieve_full(user_prompt, [m[1] for m in summary_memories])

        full_context = []
        if retrieval_decision:
            full_memories = self.memory.retrieve_memories(prompt_embedding, 'F')
            full_context = [m[1].content for m in full_memories]

        # Build context
        context = [
            "Relevant summaries:",
            *[m[1].content for m in summary_memories],
            *(["Full context:", *full_context] if full_context else []),
            "Current prompt: " + user_prompt
        ]

        # Generate response
        response = gemini_model.generate_content("\n\n".join(context))

        # Post-processing
        self._store_conversation_pair(user_prompt, response.text)

        return response.text

    def _should_retrieve_full(self, prompt, summary_memories):
        if not summary_memories:
            return False

        decision_prompt = f"""Should we retrieve full conversation history?
Current prompt: {prompt}
Summaries: {[m.content for m in summary_memories]}
Respond ONLY with 'YES' or 'NO'."""

        response = gemini_model.generate_content(decision_prompt)
        return "YES" in response.text.upper()

    def _store_conversation_pair(self, prompt, response):
        # Store full pair
        full_text = f"User: {prompt}\nAgent: {response}"
        self.memory.store_memory(full_text, 'F')

        # Create and store summary
        summary = gemini_model.generate_content(self.summarizer_prompt + full_text)
        self.memory.store_memory(summary.text, 'S')


def main():
    # Initialize the agentic system
    agent = AgenticSystem()

    # First interaction
    print("=== First Response ===")
    response1 = agent.generate_response("Hello. Poem about the meaning of life in 2 verses.")
    print(response1)

    # Second interaction
    print("\n=== Second Response ===")
    response2 = agent.generate_response("Now make it more hopeful, using ocean metaphors.")
    print(response2)


if __name__ == "__main__":
    main()