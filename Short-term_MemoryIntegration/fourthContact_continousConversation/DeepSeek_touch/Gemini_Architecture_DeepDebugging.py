import faiss
import os
import numpy as np
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
from typing import List, Tuple

# Configuration - Using Inner Product (cosine similarity)
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
        """Retrieve memories using LLM-based ranking"""
        # Get memories of the specified type
        type_memories = [m for m in self.memories if m.type == memory_type]

        if not type_memories:
            return []

        summaries = [m.content for m in type_memories]

        # Use index to get initial candidates
        index = FAISS_INDEX_S if memory_type == 'S' else FAISS_INDEX_F
        if index.ntotal == 0:
            return []

        D, I = index.search(np.array([query_embedding]), min(top_k, index.ntotal))

        # Map indices back to memory items and filter by similarity threshold
        results = []
        for i, (dist, idx) in enumerate(zip(D[0], I[0])):
            if dist > min_similarity and idx < len(type_memories):
                memory = type_memories[idx]
                results.append((dist, memory))

        # Debug: Show ranked results
        print(f"\n=== Ranked {memory_type} Memories ===")
        for score, memory in results:
            print(f"Memory ID {memory.id} - Score: {score:.4f} - Content: {memory.content}")

        return results


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
        self.conversation_history = []

    @staticmethod
    def rank_memory_summaries(query_embedding: np.ndarray, summaries: List[str]) -> List[Tuple[float, str]]:
        """Rank memory summaries using LLM-based attention mechanism"""
        if not summaries:
            return []

        # Get the query text from embedding for the prompt
        # Since we can't go back from embedding to text, we'll use a placeholder for this function
        # In real implementation, you'd want to pass the original query text
        query_text = "current user query"  # This is a placeholder

        attention_prompt = (
            f"Rank the following memory summaries based on their relevance to the user's current prompt: {query_text}\n"
            "Carefully consider the following criteria for ranking:\n"
            "- How well does the summary capture the core themes and concepts of the user's prompt?\n"
            "- Does the summary reflect the emotional tone and intent expressed in the prompt?\n"
            "- Does the summary include any key themes, entities, or metaphors mentioned in the prompt?\n"
            "- Does the summary directly or subtly address any open-ended questions or implicit requests in the prompt?\n"
            "- Is there an implicit need for continuation from previous interactions that is subtly hinted at in the prompt?\n"
            "- Consider any **implicit context** that may need to be carried over (e.g. a request for elaboration on a topic).\n"
            "- Rank higher those summaries that maintain or extend the ongoing **conversation thread** or **expand upon previous ideas**, even if they do not directly repeat specific words.\n"
            "- If the user seems to ask for a **continuation** or **variation** of a prior idea (e.g., 'Now make it more better' or 'Use X'), rank summaries that reflect this shift or expansion more highly."
        )

        for i, summary in enumerate(summaries):
            attention_prompt += f"\nSummary {i + 1}: {summary}"

        attention_prompt += "\n\nReturn a list of scores for each summary from 0 to 1, with higher scores indicating greater relevance."

        try:
            response = gemini_model.generate_content(attention_prompt)
            # Parse the scores from the response
            scores = []
            for line in response.text.split('\n'):
                if ':' in line:
                    parts = line.split(':')
                    if len(parts) == 2 and parts[1].strip().replace('.', '', 1).isdigit():
                        scores.append(float(parts[1].strip()))
                else:
                    for word in line.split():
                        if word.replace('.', '', 1).isdigit():
                            scores.append(float(word))

            # If we couldn't parse scores, use cosine similarity as fallback
            if not scores or len(scores) != len(summaries):
                print("Failed to parse LLM ranking scores, using cosine similarity fallback")
                scores = []
                for summary in summaries:
                    summary_embedding = EMBEDDING_MODEL.encode(summary)
                    score = np.dot(query_embedding, summary_embedding) / (
                            np.linalg.norm(query_embedding) * np.linalg.norm(summary_embedding)
                    )
                    scores.append(score)

            return sorted(zip(scores, summaries), reverse=True)
        except Exception as e:
            print(f"Error in rank_memory_summaries: {e}")
            return []

    def generate_response(self, user_prompt: str) -> str:
        print(f"\n{'=' * 30}\nProcessing: {user_prompt}\n{'=' * 30}")

        # Add to conversation history
        self.conversation_history.append(f"User: {user_prompt}")

        # Phase 1: Prompt Embedding
        prompt_embedding = EMBEDDING_MODEL.encode(user_prompt)

        # Phase 2: Summary Memory Retrieval
        print(f"\n[Phase 1] Summary Memory Retrieval")
        summary_memories = self.memory.retrieve_memories(
            prompt_embedding, 'S',
            top_k=5,
            min_similarity=self.similarity_threshold
        )
        print(f"Found {len(summary_memories)} relevant summaries")

        # Phase 3: Relevance Decision Making
        print("\n[Phase 2] Relevance Decision")
        retrieval_decision = self._should_retrieve_full(
            user_prompt,
            [m[1] for m in summary_memories]
        )
        print(f"Full retrieval needed? {retrieval_decision}")

        # Phase 4: Detailed Memory Retrieval
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

        # Add to conversation history
        response_text = response.text
        self.conversation_history.append(f"Agent: {response_text}")

        # Store Interaction
        self._store_conversation_pair(user_prompt, response_text)

        return response_text

    def _should_retrieve_full(self, prompt: str, summary_memories: List) -> bool:
        """LLM-based relevance decision maker"""
        if not summary_memories:
            return False

        decision_prompt = f"""Should we retrieve detailed conversation history based on:
Current prompt: {prompt}
Available summaries: {[m.content for m in summary_memories]}
Respond ONLY with 'YES' or 'NO'."""

        try:
            response = gemini_model.generate_content(decision_prompt)
            return "YES" in response.text.upper()
        except Exception as e:
            print(f"Error in _should_retrieve_full: {e}")
            return False

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

        # Add instruction for the LLM to respond naturally
        context.append(
            "\n## Instructions\nRespond naturally to the user's query using the provided context. Keep your tone conversational and helpful.")

        return context

    def _store_conversation_pair(self, prompt: str, response: str):
        """Store conversation pair with proper indexing"""
        full_text = f"User: {prompt}\nAgent: {response}"
        self.memory.store_memory(full_text, 'F')

        # Generate and store summary
        try:
            summary_prompt = self.summarizer_prompt + "\n\n" + full_text
            summary = gemini_model.generate_content(summary_prompt)
            self.memory.store_memory(summary.text, 'S')
        except Exception as e:
            print(f"Error generating summary: {e}")
            # Fallback: Use the first part of the conversation as a summary
            fallback_summary = f"User asked about {prompt[:50]}..." if len(prompt) > 50 else prompt
            self.memory.store_memory(fallback_summary, 'S')


def main():
    """Interactive conversation loop with the agent"""
    agent = AgenticSystem()
    print("=" * 50)
    print("Welcome to the Continuous Conversation Agent!")
    print("Type 'quit', 'exit', or 'bye' to end the conversation.")
    print("=" * 50)

    while True:
        user_input = input("\nYou: ")

        # Check for exit commands
        if user_input.lower() in ['quit', 'exit', 'bye']:
            print("\nThank you for chatting! Goodbye!")
            break

        # Generate and display response
        try:
            response = agent.generate_response(user_input)
            print(f"\nAgent: {response}")
        except Exception as e:
            print(f"\nError: {e}")
            print("Sorry, I encountered an error. Let's continue our conversation.")


if __name__ == "__main__":
    main()