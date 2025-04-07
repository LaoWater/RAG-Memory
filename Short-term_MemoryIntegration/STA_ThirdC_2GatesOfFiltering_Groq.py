# Third Contact.. But going further into Modelling the Mind -
# the Mind after pondering on type 'S' if it should retrieve full memory, it does not retrieve all type 'F' k neighbours, but rather chunks of them.
# So - in Third contact we shift towards a deeper memory retrieval system with relevance filtering at both the memory and chunk levels:
# Now, slowly, we're getting closer to the great Model of the Mind strictly regarding Short-Term Memory
# (Long-Term needs time-dimension and decay and for H_A is not relevant with priority right now).


import faiss
import os
import numpy as np
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
from typing import List, Tuple

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

    def store_memory(self, text: str, memory_type: str, store_chunks: bool = False):
        """Store memory with optional chunk-level embeddings"""
        embedding = EMBEDDING_MODEL.encode(text)
        item = self.MemoryItem(
            memory_id=self.current_id,
            content=text,
            embedding=embedding,
            memory_type=memory_type
        )
        self.memories.append(item)

        # Store chunk embeddings for full memories
        if store_chunks and memory_type == 'F':
            chunks = [c.strip() for c in text.split('\n') if c.strip()]
            for chunk in chunks:
                chunk_embedding = EMBEDDING_MODEL.encode(chunk)
                FAISS_INDEX_F.add(np.array([chunk_embedding]))

        # Add to main index
        if memory_type == 'S':
            FAISS_INDEX_S.add(np.array([embedding]))
        else:
            FAISS_INDEX_F.add(np.array([embedding]))

        self.current_id += 1

    def retrieve_memories(self, query_embedding: np.ndarray, memory_type: str,
                          top_k: int = 3, min_similarity: float = 0.2) -> List[Tuple[float, 'MemoryItem']]:
        """Retrieve memories with similarity filtering"""
        scores, indices = self._search_index(query_embedding, memory_type, top_k)

        results = []
        if indices.shape[0] == 0:  # No results
            return results


        for i in range(top_k):
            if indices[0][i] < 0:  # FAISS uses -1 for empty slots
                continue

            similarity = 1 - scores[0][i]  # Convert L2 distance to similarity
            idx = indices[0][i]

            if idx < len(self.memories) and similarity >= min_similarity:
                results.append((similarity, self.memories[idx]))

        return sorted(results, key=lambda x: x[0], reverse=True)

    @staticmethod
    def _search_index(query_embedding: np.ndarray, memory_type: str, k: int = 5):
        """Safe index search with empty index handling"""
        index = FAISS_INDEX_S if memory_type == 'S' else FAISS_INDEX_F
        if index.ntotal == 0:
            return np.empty((0, k)), np.empty((0, k), dtype=int)
        return index.search(np.array([query_embedding]), k)


class AgenticSystem:
    def __init__(self):
        self.memory = ConversationMemory()
        self.similarity_threshold = 0.55
        self.summarizer_prompt = """Please create a concise summary of this conversation pair that preserves:
- Core concepts
- Emotional tone
- Key entities
- Open questions
Keep under 3 sentences. Return only the summary."""

    def generate_response(self, user_prompt: str) -> str:
        # Embed the user prompt
        prompt_embedding = EMBEDDING_MODEL.encode(user_prompt)

        # Phase 1: Summary Memory Retrieval
        summary_memories = self.memory.retrieve_memories(
            prompt_embedding, 'S',
            top_k=5,
            min_similarity=self.similarity_threshold + 0.1
        )

        # Phase 2: Relevance Decision Making
        retrieval_decision = self._should_retrieve_full(
            user_prompt,
            [m[1] for m in summary_memories]
        )

        # Phase 3: Detailed Memory Retrieval
        full_context = []
        if retrieval_decision:
            raw_full_memories = self.memory.retrieve_memories(
                prompt_embedding, 'F',
                top_k=10,
                min_similarity=self.similarity_threshold - 0.1
            )
            full_context = self._filter_memory_chunks(
                user_prompt,
                prompt_embedding,
                raw_full_memories
            )

        # Build Context Hierarchy
        context = self._build_context(user_prompt, summary_memories, full_context)

        # Generate Response
        response = gemini_model.generate_content("\n\n".join(context))

        # Store Interaction
        self._store_conversation_pair(user_prompt, response.text)

        return response.text

    @staticmethod
    def _should_retrieve_full(prompt: str, summary_memories: List) -> bool:
        """LLM-based relevance decision maker"""
        if not summary_memories:
            return False

        decision_prompt = f"""Should we retrieve detailed conversation history based on:
Current prompt: {prompt}
Available summaries: {[m.content for m in summary_memories]}
Respond ONLY with 'YES' or 'NO'."""

        response = gemini_model.generate_content(decision_prompt)
        return "YES" in response.text.upper()

    def _filter_memory_chunks(self, query: str, query_embedding: np.ndarray,
                              full_memories: List[Tuple[float, ConversationMemory.MemoryItem]]) -> List[str]:
        """Chunk-level relevance filtering"""
        relevant_chunks = []

        for score, memory in full_memories:
            chunks = [
                chunk.strip()
                for chunk in memory.content.split('\n')
                if chunk.strip()
            ]

            for chunk in chunks:
                chunk_embedding = EMBEDDING_MODEL.encode(chunk)
                similarity = np.dot(query_embedding, chunk_embedding) / (
                        np.linalg.norm(query_embedding) *
                        np.linalg.norm(chunk_embedding)
                )

                if similarity > self.similarity_threshold:
                    relevant_chunks.append((
                        similarity,
                        f"[Relevance: {similarity:.2f}] {chunk}"
                    ))

        # Return top 3 most relevant chunks
        return [c[1] for c in sorted(relevant_chunks, reverse=True)[:3]]

    def _build_context(self, prompt: str, summary_memories: List,
                       full_chunks: List[str]) -> List[str]:
        """Structured context assembly"""
        context = []

        if summary_memories:
            context.append("## High-Level Context (Summaries)")
            context += [
                f"Summary {i + 1} [Confidence: {m[0]:.2f}]: {m[1].content}"
                for i, m in enumerate(summary_memories[:3])  # Top 3 summaries
            ]

        if full_chunks:
            context.append("\n## Detailed Context (Relevant Excerpts)")
            context += full_chunks[:3]  # Top 3 chunks

        context.append(f"\n## Current Query\n{prompt}")
        return context

    def _store_conversation_pair(self, prompt: str, response: str):
        """Store conversation with chunk-aware embedding"""
        full_text = f"User: {prompt}\nAgent: {response}"
        self.memory.store_memory(full_text, 'F', store_chunks=True)

        # Generate and store summary
        summary = gemini_model.generate_content(self.summarizer_prompt + full_text)
        self.memory.store_memory(summary.text, 'S')


def main():
    agent = AgenticSystem()

    print("=== First Interaction ===")
    response1 = agent.generate_response("Hello. Poem about the meaning of life in 2 verses.")
    print(f"Response:\n{response1}\n")

    print("=== Second Interaction ===")
    response2 = agent.generate_response("Now make it more hopeful, using ocean metaphors.")
    print(f"Response:\n{response2}\n")

    print("=== Third Interaction ===")
    response3 = agent.generate_response("Add a stanza about resilience against storms.")
    print(f"Response:\n{response3}")


if __name__ == "__main__":
    main()

