import os
from dotenv import load_dotenv
from groq import Groq
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.llms.groq import Groq as GroqLLM

# Load environment variables
load_dotenv()


class GroqInference:
    def __init__(self, model_name: str):
        self.client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
        self.model_name = model_name

    def generate(self, prompt: str) -> str:
        chat_completion = self.client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=self.model_name,
            temperature=0.1,
            max_tokens=1024
        )
        return chat_completion.choices[0].message.content


def main():
    # Initialize Groq with Llama3-70B
    groq_llm = GroqInference(model_name="llama3-70b-8192")

    # Configure LlamaIndex settings
    Settings.llm = GroqLLM(
        model="llama3-70b-8192",
        api_key=os.environ.get("GROQ_API_KEY"),
        temperature=0.1,
        max_tokens=1024
    )

    # Load data from data.txt in current directory
    documents = SimpleDirectoryReader(input_files=["data.txt"]).load_data()

    # Create index
    index = VectorStoreIndex.from_documents(documents)

    # Create query engine
    query_engine = index.as_query_engine()

    # Perform query using document context
    query = "Based on the document, write a philosophical poem about the meaning of life in two verses."
    response = query_engine.query(query)

    print("Context-based Poem:")
    print(response.response)

    # # Direct inference example without context
    # direct_response = groq_llm.generate("Briefly explain quantum entanglement in relation to Coconuts.")
    # print("\nDirect Inference Response:")
    # print(direct_response)


if __name__ == "__main__":
    main()

