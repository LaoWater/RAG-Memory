"""
Continuous‑Conversation Agent
Now powered by Groq Llama‑4‑Scout‑17B instead of Gemini‑1.5‑Flash
----------------------------------------------------------------
• Windows‑friendly (uses setx GROQ_API_KEY …)
• Retrieval‑augmented with FAISS + all‑mpnet‑base‑v2 embeddings
"""

import os
import faiss
import numpy as np
from typing import List, Tuple
from sentence_transformers import SentenceTransformer
from groq import Groq  # NEW ─────────────────────────────────

# ────────────────────────────── Embedding / FAISS config ──
EMBEDDING_MODEL = SentenceTransformer('all-mpnet-base-v2')
EMBEDDING_SIZE = 768
FAISS_INDEX_S = faiss.IndexFlatIP(EMBEDDING_SIZE)  # summaries
FAISS_INDEX_F = faiss.IndexFlatIP(EMBEDDING_SIZE)  # full chats

# ────────────────────────────── Groq helper ───────────────
groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))


def groq_generate(prompt: str,
                  model: str = "meta-llama/llama-4-scout-17b-16e-instruct",
                  system: str | None = None) -> str:
    """Call Groq Chat Completion and return the assistant message text."""
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    completion = groq_client.chat.completions.create(
        messages=messages,
        model=model,
    )
    return completion.choices[0].message.content


# ───────────────────────────────────────────────────────────


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

    # ---------- storage -------------------------------------------------
    def store_memory(self, text: str, memory_type: str):
        embedding = EMBEDDING_MODEL.encode(text)
        item = self.MemoryItem(self.current_id, text, embedding, memory_type)
        self.memories.append(item)

        print(f"\n=== Storing {memory_type} Memory [ID:{self.current_id}] ===")
        print(f"Content: {text}")
        print(f"Embedding‑norm: {np.linalg.norm(embedding):.4f}")

        if memory_type == 'S':
            FAISS_INDEX_S.add(np.array([embedding]))
            print(f"Added to S‑index (size: {FAISS_INDEX_S.ntotal})")
        else:
            FAISS_INDEX_F.add(np.array([embedding]))
            print(f"Added to F‑index (size: {FAISS_INDEX_F.ntotal})")

        self.current_id += 1

    # ---------- retrieval ----------------------------------------------
    def retrieve_memories(self, query_embedding: np.ndarray, memory_type: str,
                          top_k: int = 3, min_similarity: float = 0.2
                          ) -> List[Tuple[float, 'ConversationMemory.MemoryItem']]:
        type_memories = [m for m in self.memories if m.type == memory_type]
        if not type_memories:
            return []

        index = FAISS_INDEX_S if memory_type == 'S' else FAISS_INDEX_F
        if index.ntotal == 0:
            return []

        D, I = index.search(np.array([query_embedding]), min(top_k, index.ntotal))

        results = []
        for dist, idx in zip(D[0], I[0]):
            if dist > min_similarity and idx < len(type_memories):
                results.append((dist, type_memories[idx]))

        print(f"\n=== Ranked {memory_type} Memories ===")
        for score, mem in results:
            print(f"ID {mem.id} | score {score:.4f} | {mem.content[:70]}")

        return results


class AgenticSystem:
    def __init__(self):
        self.memory = ConversationMemory()
        self.similarity_threshold = 0.44
        self.summarizer_prompt = (
            "Please create a concise summary of this conversation pair that preserves:\n"
            "- Core concepts\n- Emotional tone\n- Key entities\n- Open questions\n"
            "Keep under 3 sentences. Return only the summary."
        )
        self.conversation_history = []

    # ---------- utility: rank summaries -------------------------------
    @staticmethod
    def rank_memory_summaries(query_embedding: np.ndarray,
                              summaries: List[str]) -> List[Tuple[float, str]]:
        if not summaries:
            return []

        query_text = "current user query"  # cannot invert embedding
        attention_prompt = (
            f"Rank the following memory summaries for relevance to: {query_text}\n"
            "Return one floating score 0‑1 per line in the same order."
        )
        for i, sm in enumerate(summaries, 1):
            attention_prompt += f"\n{i}. {sm}"

        response = groq_generate(attention_prompt)

        # parse any floats that appear
        scores = []
        for token in response.split():
            try:
                scores.append(float(token))
            except ValueError:
                pass
        if len(scores) != len(summaries):
            # fallback to cosine similarity
            scores = []
            for sm in summaries:
                sm_emb = EMBEDDING_MODEL.encode(sm)
                sim = np.dot(query_embedding, sm_emb) / (
                        np.linalg.norm(query_embedding) * np.linalg.norm(sm_emb))
                scores.append(sim)

        return sorted(zip(scores, summaries), reverse=True)

    # ------------------------------------------------------------------
    def _should_retrieve_full(self, prompt: str, summary_memories) -> bool:
        if not summary_memories:
            return False

        decision_prompt = (
            f"Current prompt: {prompt}\n"
            f"Summaries: {[m.content for m in summary_memories]}\n"
            "Reply ONLY YES or NO: should the agent retrieve detailed memories?"
        )
        response = groq_generate(decision_prompt)
        return "YES" in response.upper()

    # ------------------------------------------------------------------
    def _filter_memory_chunks(self, query: str, query_embedding: np.ndarray,
                              full_memories) -> List[str]:
        relevant = []
        qnorm = np.linalg.norm(query_embedding)

        for score, mem in full_memories:
            chunks = [c.strip() for c in mem.content.split('\n') if c.strip()]
            for ch in chunks:
                ch_emb = EMBEDDING_MODEL.encode(ch)
                sim = np.dot(query_embedding, ch_emb) / (qnorm * np.linalg.norm(ch_emb))
                if sim > self.similarity_threshold:
                    relevant.append((sim, f"[Sim {sim:.2f}] {ch}"))

        return [c[1] for c in sorted(relevant, reverse=True)[:3]]

    # ------------------------------------------------------------------
    def _build_context(self, prompt: str, summary_memories, full_chunks) -> List[str]:
        ctx = []
        if summary_memories:
            ctx.append("## High‑Level Context")
            ctx += [f"• {m[1].content}" for m in summary_memories[:3]]
        if full_chunks:
            ctx.append("## Detailed Excerpts")
            ctx += full_chunks
        ctx.append("## Current Query\n" + prompt)
        ctx.append("## Instructions\nRespond helpfully and conversationally.")
        return ctx

    # ------------------------------------------------------------------
    def _store_conversation_pair(self, prompt: str, response: str):
        full_text = f"User: {prompt}\nAgent: {response}"
        self.memory.store_memory(full_text, 'F')

        try:
            summary_prompt = self.summarizer_prompt + "\n\n" + full_text
            summary = groq_generate(summary_prompt)
            self.memory.store_memory(summary, 'S')
        except Exception as e:
            print("Summary gen failed:", e)
            self.memory.store_memory(prompt[:80], 'S')

    # ------------------------------------------------------------------
    def generate_response(self, user_prompt: str) -> str:
        print(f"\n{'=' * 30}\nProcessing: {user_prompt}\n{'=' * 30}")
        self.conversation_history.append("User: " + user_prompt)

        prompt_emb = EMBEDDING_MODEL.encode(user_prompt)

        # Phase 1: retrieve summaries
        summaries = self.memory.retrieve_memories(
            prompt_emb, 'S', top_k=5, min_similarity=self.similarity_threshold)
        # decide if need full
        need_full = self._should_retrieve_full(user_prompt, [m[1] for m in summaries])
        full_chunks = []
        if need_full:
            raw_full = self.memory.retrieve_memories(
                prompt_emb, 'F', top_k=10, min_similarity=self.similarity_threshold - 0.1)
            full_chunks = self._filter_memory_chunks(user_prompt, prompt_emb, raw_full)

        context = self._build_context(user_prompt, summaries, full_chunks)

        response_text = groq_generate("\n\n".join(context))
        self.conversation_history.append("Agent: " + response_text)

        self._store_conversation_pair(user_prompt, response_text)
        return response_text


# ──────────────────────────────── CLI loop ────────────────────────────
def main():
    agent = AgenticSystem()
    print("=" * 55)
    print(" Continuous Conversation Agent (Groq‑powered) ")
    print("  type 'quit' to exit")
    print("=" * 55)
    while True:
        user = input("\nYou: ")
        if user.lower() in {"quit", "exit", "bye"}:
            break
        try:
            reply = agent.generate_response(user)
            print("\nAgent:", reply)
        except Exception as err:
            print("Error:", err)


if __name__ == "__main__":
    main()
