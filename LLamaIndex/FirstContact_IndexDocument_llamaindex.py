import os
import google.generativeai as genai
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, ServiceContext
from llama_index.core.llms.base import LLM
from llama_index.core.llms.types import CompletionResponse

#####################################################################
## First Contact: Deprecated Libraries use, incomplete Wrappers #####
#####################################################################
# Step 1: Setup Gemini
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
gemini_model = genai.GenerativeModel('gemini-1.5-flash')


# Step 2: Custom Gemini wrapper for LlamaIndex
class GeminiLLM(LLM):
    def complete(self, prompt, **kwargs) -> CompletionResponse:
        response = gemini_model.generate_content(prompt)
        return CompletionResponse(text=response.text)


# Step 3: Load your dummy document
documents = SimpleDirectoryReader("./").load_data()

# Step 4: Set up LlamaIndex with Gemini
service_context = ServiceContext.from_defaults(llm=GeminiLLM())
index = VectorStoreIndex.from_documents(documents, service_context=service_context)

# Step 5: Ask it questions
query_engine = index.as_query_engine()
response = query_engine.query("Why are coconuts important in AI development?")
print(response)
