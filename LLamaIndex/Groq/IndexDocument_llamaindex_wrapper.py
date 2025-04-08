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
"""

import os
import asyncio
from typing import List, Optional, Sequence, Any
from groq import Groq
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, ServiceContext
from llama_index.core.base.llms.types import ChatResponseGen
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

# Ensure GROQ_API_KEY is set
if "GROQ_API_KEY" not in os.environ:
    raise ValueError("Please set GROQ_API_KEY in your environment or .env file")


class GroqLLM(LLM):
    """Custom LLM implementation that uses Groq's API."""

    async def astream_chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponseAsyncGen:
        pass

    def __init__(
            self,
            model: str = "mixtral-8x7b-32768",
            temperature: float = 0.2,
            max_tokens: int = 512
    ):
        super().__init__()
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

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


def load_documents(data_dir: str = "data"):
    """Load documents from a directory."""
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        print(f"Created directory: {data_dir}")
        print(f"Please add your documents to the {data_dir} directory.")
        return None

    return SimpleDirectoryReader(data_dir).load_data()


def setup_index(documents, llm):
    """Set up the vector store index with the provided LLM."""
    service_context = ServiceContext.from_defaults(llm=llm)
    return VectorStoreIndex.from_documents(documents, service_context=service_context)


def query_documents(index, query_text: str):
    """Query the index with the provided text."""
    query_engine = index.as_query_engine()
    return query_engine.query(query_text)


def main():
    # Initialize the Groq LLM
    groq_llm = GroqLLM(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        temperature=0.2,
        max_tokens=1024
    )

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
    index = setup_index(documents, groq_llm)

    # Interactive query loop
    prompt = "Why are coconuts important in AI development?"
    response = query_documents(index, prompt)
    print("\nResponse:")
    print(response)
    print("-" * 50)


if __name__ == "__main__":
    main()
