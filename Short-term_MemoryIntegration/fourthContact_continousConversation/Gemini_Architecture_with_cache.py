import os
import faiss
import numpy as np
from typing import List, Tuple

import google.generativeai as genai

# FAISS Indexes
EMBEDDING_SIZE = 768
FAISS_INDEX_S = faiss.IndexFlatIP(EMBEDDING_SIZE)
FAISS_INDEX_F = faiss.IndexFlatIP(EMBEDDING_SIZE)

# Configure Gemini
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
gemini_model = genai.GenerativeModel('gemini-1.5-flash')


class ConversationMemory:
    def __init__(self, embedding_model):
        self.embedding_model = embedding_model
        self.memories = []
        self.current_id = 0

    class MemoryItem:
        def __init__(self, memory_id, content, embedding, memory_type):
            self.id = memory_id
            self.content = content
            self.embedding = embedding
            self.type = memory_type

    def store_memory(self, text: str, memory_type: str):
        embedding = self.embedding_model.encode(text)
        item = self.MemoryItem(
            memory_id=self.current_id,
            content=text,
            embedding=embedding,
            memory_type=memory_type
        )
        self.memories.append(item)

        if memory_type == 'S':
            FAISS_INDEX_S.add(np.array([embedding]))
        else:
            FAISS_INDEX_F.add(np.array([embedding]))

        self.current_id += 1

    def retrieve_memories(self, query_embedding: np.ndarray, memory_type: str,
                          top_k: int = 5, min_similarity: float = 0.2) -> List[Tuple[float, 'MemoryItem']]:
        summaries = [m.content for m in self.memories if m.type == memory_type]
        ranked = AgenticSystem.rank_memory_summaries(query_embedding, summaries)
        return [(score, self.memories[i]) for i, (score, _) in enumerate(ranked) if score > min_similarity]


class ConversationThread:
    def __init__(self):
        self.turns = []

    def append(self, user_input: str, agent_response: str):
        self.turns.append((user_input, agent_response))

    def get_recent_text(self, n=3):
        return "\n".join(f"User: {u}\nAgent: {a}" for u, a in self.turns[-n:])


class AgenticSystem:
    def __init__(self, embedding_model):
        self.embedding_model = embedding_model
        self.memory = ConversationMemory(embedding_model)
        self.similarity_threshold = 0.44
        self.summarizer_prompt = """Please create a concise summary of this conversation pair that preserves:
- Core concepts
- Emotional tone
- Key entities
- Open questions
Keep under 3 sentences. Return only the summary."""

    @staticmethod
    def rank_memory_summaries(prompt: str, summaries: List[str]) -> List[Tuple[float, str]]:
        if not summaries:
            return []

        prompt_text = (
            f"Rank the following memory summaries based on how relevant they are to the current user message:\n"
            f"User: {prompt}\n"
            + "\n".join([f"Summary {i+1}: {s}" for i, s in enumerate(summaries)]) +
            "\n\nReturn space-separated scores between 0 and 1 only."
        )
        response = gemini_model.generate_content(prompt_text)
        scores = [float(s) for s in response.text.strip().split() if s.replace('.', '', 1).isdigit()]
        return list(zip(scores, summaries))

    def generate_response(self, user_prompt: str, thread: ConversationThread) -> str:
        prompt_embedding = self.embedding_model.encode(user_prompt)

        # Step 1: Retrieve summaries
        summary_memories = self.memory.retrieve_memories(
            prompt_embedding, 'S', top_k=5, min_similarity=self.similarity_threshold + 0.1
        )
        ranked_summaries = self.rank_memory_summaries(
            user_prompt, [m[1].content for m in summary_memories]
        )
        top_summaries = [summary for score, summary in ranked_summaries if score > self.similarity_threshold]

        # Step 2: Optionally fetch full context
        full_context = []
        if self._should_retrieve_full(user_prompt, [m[1] for m in summary_memories]):
            full_memories = self.memory.retrieve_memories(
                prompt_embedding, 'F', top_k=10, min_similarity=self.similarity_threshold - 0.1
            )
            full_context = self._filter_memory_chunks(user_prompt, prompt_embedding, full_memories)

        # Step 3: Build input context
        context_parts = []
        if top_summaries:
            context_parts.append("## High-Level Context (Summaries)")
            context_parts += [f"- {s}" for s in top_summaries[:3]]

        if full_context:
            context_parts.append("\n## Relevant Excerpts")
            context_parts += full_context

        if thread.turns:
            context_parts.append("\n## Recent Conversation")
            context_parts.append(thread.get_recent_text())

        context_parts.append(f"\n## User Message\n{user_prompt}")
        full_prompt = "\n\n".join(context_parts)

        # Step 4: Generate response
        response = gemini_model.generate_content(full_prompt).text.strip()

        # Step 5: Save to memory
        self._store_conversation_pair(user_prompt, response)

        return response

    def _should_retrieve_full(self, prompt: str, summary_memories: List) -> bool:
        if not summary_memories:
            return False
        check_prompt = f"""Based on the current prompt and previous summaries, is detailed memory needed?

Prompt: {prompt}
Summaries: {[m.content for m in summary_memories]}

Reply ONLY "YES" or "NO"."""
        response = gemini_model.generate_content(check_prompt)
        return "YES" in response.text.upper()

    def _filter_memory_chunks(self, query: str, query_embedding: np.ndarray,
                              full_memories: List[Tuple[float, ConversationMemory.MemoryItem]]) -> List[str]:
        relevant = []
        query_norm = np.linalg.norm(query_embedding)

        for score, memory in full_memories:
            for chunk in memory.content.split('\n'):
                chunk = chunk.strip()
                if not chunk:
                    continue
                chunk_emb = self.embedding_model.encode(chunk)
                sim = np.dot(query_embedding, chunk_emb) / (query_norm * np.linalg.norm(chunk_emb))
                if sim > self.similarity_threshold:
                    relevant.append((sim, chunk))

        return [f"[Relevance: {sim:.2f}] {chunk}" for sim, chunk in sorted(relevant, reverse=True)[:3]]

    def _store_conversation_pair(self, prompt: str, response: str):
        full_text = f"User: {prompt}\nAgent: {response}"
        self.memory.store_memory(full_text, 'F')
        summary = gemini_model.generate_content(self.summarizer_prompt + full_text)
        self.memory.store_memory(summary.text.strip(), 'S')


def main():
    from sentence_transformers import SentenceTransformer
    print("Loading embedding model... (this only happens once and is cached)")

    embedding_model = SentenceTransformer('all-mpnet-base-v2')  # Lazy load and cached

    agent = AgenticSystem(embedding_model)
    thread = ConversationThread()

    print("✅ Agent ready. Type your message. Type 'exit' to quit.\n")

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in {"exit", "quit"}:
            print("👋 Goodbye!")
            break

        response = agent.generate_response(user_input, thread)
        thread.append(user_input, response)
        print(f"Agent: {response}\n")


if __name__ == "__main__":
    main()
