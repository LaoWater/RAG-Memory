import os
import google.generativeai as genai
from colorama import init, Fore, Style

# Initialize colorama for Windows terminal
init()


def configure_gemini(model):
    # Always try to configure Gemini if the key exists
    gemini_api_key = os.getenv("GOOGLE_API_KEY")
    if gemini_api_key:
        try:
            genai.configure(api_key=gemini_api_key)
            print("Gemini API configured successfully.")
            # Return the Gemini model object if successful
            return genai.GenerativeModel(f'models/{model}')  # Use latest flash model
        except Exception as e:
            print(f"Gemini setup failed: {e}")
            # Fall through to return None if Gemini setup fails
    else:
        print("GOOGLE_API_KEY not found. Gemini model cannot be configured.")

    # Return None if Gemini key is missing or configuration failed
    print("Gemini model not configured.")
    return None


gemini_model = configure_gemini(model='gemini-2.5-pro-exp-03-25')
print('/n')

prompt = f"Hello. Poem about the meaning of life in 2 verses."

# Generate content
response = gemini_model.generate_content(prompt)

# Extract the text from the response
text = response.parts[0].text

print(text)

# Print with formatting
lines = text.splitlines()
