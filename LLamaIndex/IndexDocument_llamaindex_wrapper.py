import os
import google.generativeai as genai
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Settings
# Corrected import for LLM base class
from llama_index.core.llms import LLM
# Imports needed for the LLM implementation
from llama_index.core.llms.callbacks import llm_completion_callback, llm_chat_callback
from llama_index.core.llms.types import (
    ChatMessage,
    ChatResponse,
    ChatResponseGen,
    CompletionResponse,
    CompletionResponseGen,
    LLMMetadata,
    MessageRole,
)
from typing import Any, Sequence, Optional # Added Optional

# Step 1: Setup Gemini
# Ensure your GOOGLE_API_KEY environment variable is set
# Or configure directly: genai.configure(api_key="YOUR_API_KEY")
google_api_key = os.getenv("GOOGLE_API_KEY")
if not google_api_key:
    raise ValueError("GOOGLE_API_KEY environment variable not set.")
genai.configure(api_key=google_api_key)

# Choose your Gemini model
# Make sure the chosen model supports the required generation methods
# gemini-1.5-flash is a good choice
gemini_model_name = 'gemini-1.5-flash' # Or 'gemini-pro' etc.
gemini_model = genai.GenerativeModel(gemini_model_name)


# Step 2: Updated Custom Gemini wrapper for LlamaIndex
class GeminiLLM(LLM):
    # Define required metadata
    @property
    def metadata(self) -> LLMMetadata:
        """Get LLM metadata."""
        # You might want to fetch context window size etc. from Gemini API if available
        return LLMMetadata(
            context_window=8192, # Example value, check Gemini docs for actual limits
            num_output=2048,     # Example value
            is_chat_model=True, # Gemini models are typically chat models
            model_name=gemini_model_name,
        )

    # Implement the core chat method
    @llm_chat_callback()
    def chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        # Convert LlamaIndex ChatMessages to Gemini format (usually a list of dicts)
        # Note: Gemini API might have specific role names ('user', 'model')
        gemini_messages = []
        for msg in messages:
            role = "user" if msg.role == MessageRole.USER else "model"
            # Handle system prompts if necessary (Gemini might treat them differently)
            if msg.role == MessageRole.SYSTEM:
                 # Gemini API often takes system instructions separately or merged with user msg
                 # For simplicity here, let's prepend it to the next user message content
                 # or handle it according to specific Gemini best practices.
                 # This basic implementation might just add it as a 'user' turn.
                 # A better approach might use specific API features if available.
                 print(f"Warning: System message basic handling: {msg.content}")
                 gemini_messages.append({"role": "user", "parts": [msg.content]})
                 continue # Skip adding separately if handled above or ignored
            gemini_messages.append({"role": role, "parts": [msg.content]})

        # Handle potential empty messages list after filtering system prompt
        if not gemini_messages:
             return ChatResponse(message=ChatMessage(role=MessageRole.ASSISTANT, content=""))


        # Make the API call
        # print(f"Sending to Gemini: {gemini_messages}") # Debugging
        try:
            response = gemini_model.generate_content(gemini_messages)
            # print(f"Received from Gemini: {response.text}") # Debugging

            # Check for safety ratings or blocks if needed
            if not response.candidates or not response.candidates[0].content.parts:
                 # Handle cases where generation might be blocked or empty
                 print(f"Warning: Gemini response might be empty or blocked. Response: {response}")
                 assistant_content = "Generation failed or was blocked."
                 # You might want to check `response.prompt_feedback` for block reasons
            else:
                 assistant_content = response.text # Access text directly

            return ChatResponse(
                message=ChatMessage(role=MessageRole.ASSISTANT, content=assistant_content),
                raw=response.__dict__ # Store raw response if needed
            )
        except Exception as e:
             print(f"Error during Gemini API call: {e}")
             # Handle exceptions gracefully
             return ChatResponse(
                 message=ChatMessage(role=MessageRole.ASSISTANT, content=f"Error: {e}")
             )


    # Implement streaming chat (optional but good practice)
    @llm_chat_callback()
    def stream_chat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponseGen:
        # Similar conversion as in chat()
        gemini_messages = []
        for msg in messages:
            role = "user" if msg.role == MessageRole.USER else "model"
            if msg.role == MessageRole.SYSTEM:
                 print(f"Warning: System message basic handling in stream: {msg.content}")
                 gemini_messages.append({"role": "user", "parts": [msg.content]})
                 continue
            gemini_messages.append({"role": role, "parts": [msg.content]})

        if not gemini_messages:
             def empty_gen():
                  yield ChatResponse(message=ChatMessage(role=MessageRole.ASSISTANT, content=""))
             return empty_gen()


        # Make the streaming API call
        try:
            stream = gemini_model.generate_content(gemini_messages, stream=True)

            def gen() -> ChatResponseGen:
                content = ""
                role = MessageRole.ASSISTANT
                for chunk in stream:
                    # print(f"Stream chunk: {chunk.text}") # Debugging
                    delta = chunk.text
                    content += delta
                    yield ChatResponse(
                        message=ChatMessage(role=role, content=content),
                        delta=delta,
                        raw=chunk.__dict__
                    )

            return gen()
        except Exception as e:
             print(f"Error during Gemini streaming API call: {e}")
             def error_gen() -> ChatResponseGen:
                    yield ChatResponse(
                        message=ChatMessage(role=MessageRole.ASSISTANT, content=f"Error: {e}"),
                        delta=f"Error: {e}"
                    )
             return error_gen()


    # Implement complete/stream_complete (often defaults work if chat is implemented)
    # You might need to implement these if specific components rely on them.
    # The default implementation usually converts the prompt to a user message
    # and calls chat/stream_chat.

    # @llm_completion_callback()
    # def complete(self, prompt: str, formatted: bool = False, **kwargs: Any) -> CompletionResponse:
    #     # Basic implementation: Treat prompt as a single user message
    #     user_message = ChatMessage(role=MessageRole.USER, content=prompt)
    #     chat_response = self.chat([user_message], **kwargs)
    #     return CompletionResponse(text=chat_response.message.content, raw=chat_response.raw)

    # @llm_completion_callback()
    # def stream_complete(self, prompt: str, formatted: bool = False, **kwargs: Any) -> CompletionResponseGen:
    #      user_message = ChatMessage(role=MessageRole.USER, content=prompt)
    #      stream_chat_response_gen = self.stream_chat([user_message], **kwargs)

    #      def gen() -> CompletionResponseGen:
    #            full_text = ""
    #            for resp in stream_chat_response_gen:
    #                 full_text = resp.message.content # Accumulate text from chat stream
    #                 yield CompletionResponse(text=full_text, delta=resp.delta, raw=resp.raw) # Pass delta along
    #      return gen()


# Step 3: Load your dummy document (ensure it exists)
# Create a dummy data directory if it doesn't exist
if not os.path.exists("./data"):
    os.makedirs("./data")
# Create a dummy file if it doesn't exist
dummy_file_path = "./data/dummy.txt"
if not os.path.exists(dummy_file_path):
    with open(dummy_file_path, "w") as f:
        f.write("This is a dummy document about testing LlamaIndex and Gemini.\n")
        f.write("Coconuts are not typically related to AI, but let's pretend they are for this query.")

# Load data from the specific directory
documents = SimpleDirectoryReader("./data").load_data() # Load from ./data/

# Step 4: Set up LlamaIndex using Settings
# Configure the LLM globally (or locally if needed for specific components)
Settings.llm = GeminiLLM()
# You might also want to configure embeddings, e.g., using Gemini embeddings
# Settings.embed_model = GeminiEmbedding() # If using Gemini embeddings

# Build the index using the global Settings
index = VectorStoreIndex.from_documents(documents) # No need to pass service_context

# Step 5: Ask it questions
query_engine = index.as_query_engine() # Uses the LLM from Settings
response = query_engine.query("Why are coconuts important in AI development?")
print(response)