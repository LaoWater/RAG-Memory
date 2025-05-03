# Enter - Time Decayed Memories - as conversation goes on
# # Yet this is another level not relevant to be explored now.
#

import faiss
import numpy as np
from datetime import datetime
from sentence_transformers import SentenceTransformer
from google.colab import userdata
import google.generativeai as genai

# Configuration
EMBEDDING_MODEL = SentenceTransformer('all-mpnet-base-v2')
EMBEDDING_SIZE = 768
FAISS_INDEX_S = faiss.IndexFlatL2(EMBEDDING_SIZE)
FAISS_INDEX_F = faiss.IndexFlatL2(EMBEDDING_SIZE)

# Configure Gemini
GOOGLE_API_KEY = userdata.get('GEMINI_KEY')
genai.configure(api_key=GOOGLE_API_KEY)
gemini_model = genai.GenerativeModel('gemini-1.5-flash')


def _search_index(query_embedding, memory_type, k=5):
    index = FAISS_INDEX_S if memory_type == 'S' else FAISS_INDEX_F
    scores, indices = index.search(np.array([query_embedding]), k)
    return scores[0], indices[0]


class ConversationMemory:
    def __init__(self):
        self.memories = []
        self.current_id = 0
        self.time_decay = 0.95  # Adjust based on temporal needs

    class MemoryItem:
        def __init__(self, memory_id, content, embedding, memory_type, timestamp):
            self.id = memory_id
            self.content = content
            self.embedding = embedding
            self.type = memory_type  # 'S' or 'F'
            self.timestamp = timestamp
            self.activation = 1.0

    def store_memory(self, text, memory_type):
        embedding = EMBEDDING_MODEL.encode(text)
        item = self.MemoryItem(
            memory_id=self.current_id,
            content=text,
            embedding=embedding,
            memory_type=memory_type,
            timestamp=datetime.now()
        )
        self.memories.append(item)

        # Add to FAISS index
        if memory_type == 'S':
            FAISS_INDEX_S.add(np.array([embedding]))
        else:
            FAISS_INDEX_F.add(np.array([embedding]))

        self.current_id += 1

    def retrieve_memories(self, query_embedding, memory_type, top_k=3):
        """Retrieve with temporal activation decay"""
        scores, indices = _search_index(query_embedding, memory_type, top_k * 3)

        # Apply time-based activation decay and select top
        results = []
        for score, idx in zip(scores, indices):
            memory = self.memories[idx]
            time_decay = self.time_decay ** (datetime.now() - memory.timestamp).days
            memory.activation *= time_decay
            results.append((score * memory.activation, memory))

        return sorted(results, key=lambda x: x[0], reverse=True)[:top_k]


def _should_retrieve_full(prompt, summary_memories):
    if not summary_memories:
        return False

    decision_prompt = f"""Based on the current prompt and these summaries, should we retrieve full conversation history?
                        Current prompt: {prompt}
                        Summaries: {[m.content for m in summary_memories]}
                        Respond ONLY with 'YES' or 'NO'."""

    response = gemini_model.generate_content(decision_prompt)
    return "YES" in response.text.upper()


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

        # First retrieve summaries
        summary_memories = self.memory.retrieve_memories(prompt_embedding, 'S')

        # LLM-based relevance check
        retrieval_decision = _should_retrieve_full(user_prompt, summary_memories)

        full_context = []
        if retrieval_decision:
            full_memories = self.memory.retrieve_memories(prompt_embedding, 'F')
            full_context = [m.content for _, m in full_memories]

        # Build context
        context = [
            "Relevant summaries from previous conversations:",
            *[m.content for _, m in summary_memories],
            *(["Full relevant context:", *full_context] if full_context else []),
            "Current prompt: " + user_prompt
        ]

        # Generate response
        response = gemini_model.generate_content("\n\n".join(context))

        # Post-processing
        self._store_conversation_pair(user_prompt, response.text)

        return response.text

    def _store_conversation_pair(self, prompt, response):
        # Store full pair
        full_text = f"User: {prompt}\nAgent: {response}"
        self.memory.store_memory(full_text, 'F')

        # Create and store summary
        summary = gemini_model.generate_content(self.summarizer_prompt + full_text)
        self.memory.store_memory(summary.text, 'S')

        # Usage example
        agent = AgenticSystem()


        # First interaction
        response1 = agent.generate_response("Hello. Poem about the meaning of life in 2 verses.")
        print(response1)

        # Second interaction
        response2 = agent.generate_response("Now make it more hopeful, using ocean metaphors.")
        print(response2)