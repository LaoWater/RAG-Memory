"""
LlamaIndex integration with Groq for document querying
This script demonstrates how to:
1. Load documents using LlamaIndex
2. Create a query engine using LlamaIndex
3. Use Groq's API to generate responses

Notes:
- This script uses the latest LlamaIndex structure (as of April 2025)
- Most core functionality is now under llama_index.core
- CustomLLM has been renamed to LLM

- some issues on loading GroqAPI with llama index - needing to create class with all awaited functions definitions, up to async ones.

Finally, got it to work after adapting to latest documentations in https://docs.llamaindex.ai/en/stable/module_guides/supporting_modules/service_context_migration/
- so this script uses a more general-approach on llama-index, rather than OpenAI or Gemini specific ramifications
- it's doing everything at the low-level of the library, setting model, creating index, embedings, metdata
- Even the feeling of it, it 's low-level Parsing nodes: 100%|██████████|, Generating embeddings: 100%|██████████|, etc

- Usually - all these steps are handled in the higher level use of "from llama_index.llms.groq import Groq as GroqLLM" or Gemini, etc.

"""

import os
import asyncio
from typing import List, Optional, Sequence, Any
from groq import Groq
from llama_index.core import Settings
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.embeddings import resolve_embed_model
from llama_index.core.llms import (
    LLM,
    LLMMetadata,
    CompletionResponse,
    CompletionResponseAsyncGen,
    ChatResponse,
    ChatResponseAsyncGen,
    ChatMessage,
    MessageRole,
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.openai import OpenAIEmbedding
from pydantic import Field, PrivateAttr

# Ensure GROQ_API_KEY is set
if "GROQ_API_KEY" not in os.environ:
    raise ValueError("Please set GROQ_API_KEY in your environment or .env file")


class GroqLLM(LLM):
    """Custom LLM implementation that uses Groq's API."""

    # Define Pydantic fields first
    model: str = Field(
        default="meta-llama/llama-4-scout-17b-16e-instruct",
        description="The Groq model to use"
    )
    temperature: float = Field(
        default=0.2,
        description="The temperature to use for sampling"
    )
    max_tokens: int = Field(
        default=1024,
        description="The maximum number of tokens to generate"
    )

    # Private attributes don't need validation
    _client: Any = PrivateAttr()

    async def astream_complete(self, prompt: str, formatted: bool = False, **kwargs: Any) -> CompletionResponseAsyncGen:
        pass

    async def astream_chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponseAsyncGen:
        pass

    def __init__(
            self,
            model: str = "meta-llama/llama-4-scout-17b-16e-instruct",
            temperature: float = 0.2,
            max_tokens: int = 512
    ):
        super().__init__()
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        # Initialize the Groq client using PrivateAttr
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY environment variable not set")

        self._client = Groq(api_key=api_key)

    @property
    def client(self):
        """Access the client through a property if needed"""
        return self._client

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(
            model_name=self.model,
            max_input_size=32768,
            num_output=self.max_tokens,
            context_window=32768,
            is_chat_model=True,
        )

    def chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        groq_messages = [
            {"role": msg.role.value, "content": msg.content}
            for msg in messages
        ]
        chat_completion = self.client.chat.completions.create(
            messages=groq_messages,
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            **kwargs
        )
        response = chat_completion.choices[0].message
        return ChatResponse(
            message=ChatMessage(
                role=MessageRole(response.role),
                content=response.content
            )
        )

    async def achat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self.chat,
            messages,
            **kwargs
        )

    def complete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        response = self.client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            **kwargs
        )
        return CompletionResponse(text=response.choices[0].message.content)

    async def acomplete(self, prompt: str, formatted: bool = False, **kwargs: Any) -> CompletionResponse:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self.complete,
            prompt,
            **kwargs
        )

    async def stream_complete(self, prompt: str, formatted: bool = False, **kwargs: Any) -> CompletionResponseAsyncGen:
        # Fallback to async complete
        response = await self.acomplete(prompt, **kwargs)
        yield response

    async def stream_chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponseAsyncGen:
        # Fallback to async chat
        response = await self.achat(messages, **kwargs)
        yield response


def load_documents(file_path: str = "data.txt"):
    """Load a specific text file from the current directory."""
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        print("Please make sure the file exists in the current directory.")
        return None

    # Load just the specific file
    return SimpleDirectoryReader(input_files=[file_path]).load_data()


def query_documents(index, query_text: str):
    """Query the index with the provided text."""
    query_engine = index.as_query_engine()
    return query_engine.query(query_text)


def setup_index(documents):
    """Create and return a vector index from documents."""
    index = VectorStoreIndex.from_documents(
        documents,
        # No need to pass service_context anymore
        show_progress=True
    )
    return index


def main():
    # Initialize the Groq LLM
    # Configure settings
    # Initialize the Groq LLM
    groq_llm = GroqLLM()  # Create an instance first

    # Configure settings with the instance
    Settings.llm = groq_llm  # Pass the instance, not the class
    Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")
    Settings.node_parser = SentenceSplitter(chunk_size=512, chunk_overlap=20)
    Settings.num_output = 512
    Settings.context_window = 3900

    # Usage:
    documents = load_documents("data.txt")  # Your document loading function
    index = setup_index(documents)

    # Simple test of direct inference
    if False:  # Set to True to test direct inference, outside LlamaIndex
        test_result = groq_llm.groq_inference(
            prompt="Hello. Poem about the meaning of life in 2 verses."
        )
        print("Test Result:")
        print(test_result)
        print("-" * 50)

    # Load documents
    print("Loading documents...")
    documents = load_documents()
    if not documents:
        print("No documents found. Please add documents to the data directory.")
        return

    # Create index
    print("Creating index...")
    index = setup_index(documents)

    # Interactive query loop
    prompt = "Why are coconuts important in AI development?"
    response = query_documents(index, prompt)
    print("\nResponse:")
    print(response)
    print("-" * 50)


if __name__ == "__main__":
    main()
