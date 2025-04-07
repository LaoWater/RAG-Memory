# Third Contact.. But going further into Modelling the Mind -
# the Mind after pondering on type 'S' if it should retrieve full memory, it does not retrieve all type 'F' k neighbours, but rather chunks of them.
# So - in Third contact we shift towards a deeper memory retrieval system with relevance filtering at both the memory and chunk levels:
# Now, slowly, we're getting closer to the great Model of the Mind strictly regarding Short-Term Memory
# (Long-Term needs time-dimension and decay and for H_A is not relevant with priority right now).

# Issue at first Gate - why ?
# Debugged intimately - found that scores are too low in many cases - of course - for the retrieval lacks understanding and comprehesion.
# ENTER: Attention ? - Mirroring the Mind, the retrieval, pondering what to retrieve, get full or summary - is happening with continous Attention mechanism.
# Else, the first cycle example of Generate poem & Continue in second prompt - is not high enough attention to trigger the similarily mechanism


import faiss
import os
import numpy as np
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
from typing import List, Tuple

# Configuration - Changed to Inner Product (cosine similarity)
EMBEDDING_MODEL = SentenceTransformer('all-mpnet-base-v2')
EMBEDDING_SIZE = 768
FAISS_INDEX_S = faiss.IndexFlatIP(EMBEDDING_SIZE)
FAISS_INDEX_F = faiss.IndexFlatIP(EMBEDDING_SIZE)

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
        """Store memory with normalized embeddings"""
        embedding = EMBEDDING_MODEL.encode(text)
        item = self.MemoryItem(
            memory_id=self.current_id,
            content=text,
            embedding=embedding,
            memory_type=memory_type
        )
        self.memories.append(item)

        # Debug: Track memory storage
        print(f"\n=== Storing {memory_type} Memory [ID:{self.current_id}] ===")
        print(f"Content: {text}")
        print(f"Embedding Norm: {np.linalg.norm(embedding):.4f}")

        # Add to appropriate index
        if memory_type == 'S':
            FAISS_INDEX_S.add(np.array([embedding]))
            print(f"Added to S-index (new size: {FAISS_INDEX_S.ntotal})")
        else:
            FAISS_INDEX_F.add(np.array([embedding]))
            print(f"Added to F-index (new size: {FAISS_INDEX_F.ntotal})")

        self.current_id += 1

    def retrieve_memories(self, query_embedding: np.ndarray, memory_type: str,
                          top_k: int = 3, min_similarity: float = 0.2) -> List[Tuple[float, 'MemoryItem']]:
        """Retrieve memories with cosine similarity filtering"""
        scores, indices = self._search_index(query_embedding, memory_type, top_k)
        results = []


        if indices.shape[0] == 0:  # No results
            return results

        # Debug: Show raw search results
        print(f"\n=== Retrieving {memory_type} Memories ===")
        print(f"Top {top_k} raw results:")



        for i in range(top_k):
            if indices[0][i] < 0:  # FAISS uses -1 for empty slots
                continue
            idx = indices[0][i]
            if idx >= 0:
                print(f"Index {idx} - Score: {scores[0][i]:.4f}")


        for i in range(top_k):

            similarity = scores[0][i]  # Direct use of cosine similarity
            idx = indices[0][i]

            if idx < len(self.memories) and similarity >= min_similarity:
                results.append((similarity, self.memories[idx]))
                # Debug: Show valid matches
                print(f"Valid match: ID {self.memories[idx].id} - Similarity: {similarity:.4f}")

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
        self.similarity_threshold = 0.44
        self.summarizer_prompt = """Please create a concise summary of this conversation pair that preserves:
- Core concepts
- Emotional tone
- Key entities
- Open questions
Keep under 3 sentences. Return only the summary."""

    @staticmethod
    def rank_memory_summaries(user_prompt: str, summaries: List[str]) -> List[Tuple[float, str]]:
        """Rank memory summaries using LLM-based attention mechanism"""
        attention_prompt = f"Rank the following summaries based on their relevance to the prompt: {user_prompt}\n"
        attention_prompt += "\n".join([f"Summary {i+1}: {summary}" for i, summary in enumerate(summaries)])
        attention_prompt += "\nReturn a list of scores for each summary."

        response = gemini_model.generate_content(attention_prompt)
        scores = [float(score) for score in response.text.split() if score.replace('.', '', 1).isdigit()]

        return sorted(zip(scores, summaries), reverse=True)


    def generate_response(self, user_prompt: str) -> str:
        print(f"\n{'=' * 30}\nProcessing: {user_prompt}\n{'=' * 30}")

        # Phase 1: Summary Memory Retrieval
        prompt_embedding = EMBEDDING_MODEL.encode(user_prompt)
        print(f"\n[Phase 1] Summary Memory Retrieval")
        summary_memories = self.memory.retrieve_memories(
            prompt_embedding, 'S',
            top_k=5,
            min_similarity=self.similarity_threshold + 0.1
        )
        print(f"Found {len(summary_memories)} relevant summaries")

        # Phase 2: Rank Memory Summaries
        print("\n[Phase 2] Rank Memory Summaries")
        ranked_summaries = self.rank_memory_summaries(
            user_prompt,
            [m[1].content for m in summary_memories]
        )
        print(f"Ranked summaries: {ranked_summaries}")

        # Phase 3: Retrieve Top-Ranked Summaries
        print("\n[Phase 3] Retrieve Top-Ranked Summaries")
        top_summaries = [summary for score, summary in ranked_summaries if score > self.similarity_threshold]
        print(f"Top summaries: {top_summaries}")

        # Phase 4: Relevance Decision Making
        print("\n[Phase 2] Relevance Decision")
        retrieval_decision = self._should_retrieve_full(
            user_prompt,
            [m[1] for m in summary_memories]
        )
        print(f"Full retrieval needed? {retrieval_decision}")

        # Phase 3: Detailed Memory Retrieval
        full_context = []
        if retrieval_decision:
            print("\n[Phase 3] Detailed Memory Retrieval")
            raw_full_memories = self.memory.retrieve_memories(
                prompt_embedding, 'F',
                top_k=10,
                min_similarity=self.similarity_threshold - 0.1
            )
            print(f"Found {len(raw_full_memories)} full memories")
            full_context = self._filter_memory_chunks(
                user_prompt,
                prompt_embedding,
                raw_full_memories
            )
            print(f"Selected {len(full_context)} relevant chunks")

        # Build Context Hierarchy
        context = self._build_context(user_prompt, summary_memories, full_context)

        # Generate Response
        print("\n[Generation] Using context:")
        for c in context: print(f"| {c}")
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
        query_norm = np.linalg.norm(query_embedding)

        for score, memory in full_memories:
            print(f"\nProcessing memory {memory.id}:")
            chunks = [
                chunk.strip()
                for chunk in memory.content.split('\n')
                if chunk.strip()
            ]
            print(f"Found {len(chunks)} chunks")

            for chunk in chunks:
                chunk_embedding = EMBEDDING_MODEL.encode(chunk)
                similarity = np.dot(query_embedding, chunk_embedding) / (
                        query_norm * np.linalg.norm(chunk_embedding)
                )

                if similarity > self.similarity_threshold:
                    print(f"Chunk match: {similarity:.2f} - {chunk}")
                    relevant_chunks.append((
                        similarity,
                        f"[Relevance: {similarity:.2f}] {chunk}"
                    ))

        return [c[1] for c in sorted(relevant_chunks, reverse=True)[:3]]

    def _build_context(self, prompt: str, summary_memories: List,
                       full_chunks: List[str]) -> List[str]:
        """Structured context assembly"""
        context = []
        if summary_memories:
            context.append("## High-Level Context (Summaries)")
            context += [
                f"Summary {i + 1} [Confidence: {m[0]:.2f}]: {m[1].content}"
                for i, m in enumerate(summary_memories[:3])
            ]

        if full_chunks:
            context.append("\n## Detailed Context (Relevant Excerpts)")
            context += full_chunks[:3]

        context.append(f"\n## Current Query\n{prompt}")
        return context

    def _store_conversation_pair(self, prompt: str, response: str):
        """Store conversation pair with proper indexing"""
        full_text = f"User: {prompt}\nAgent: {response}"
        self.memory.store_memory(full_text, 'F')

        # Generate and store summary
        summary = gemini_model.generate_content(self.summarizer_prompt + full_text)
        self.memory.store_memory(summary.text, 'S')


def main():
    agent = AgenticSystem()

    print("=== First Interaction ===")
    response1 = agent.generate_response("Hello. Poem about the meaning of life in 2 verses.")
    print(f"\n Response:\n{response1}\n")

    print("=== Second Interaction ===")
    response2 = agent.generate_response("Now make it more hopeful, using ocean metaphors.")
    print(f"Response:\n{response2}\n")

    print("=== Third Interaction ===")
    response3 = agent.generate_response("Add a stanza about resilience against storms.")
    print(f"Response:\n{response3}")


if __name__ == "__main__":
    main()
