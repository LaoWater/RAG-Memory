import os
from groq import Groq


def groq_inference(model: str, prompt: str) -> str:
    client = Groq(
        api_key=os.environ.get("GROQ_API_KEY"),
    )

    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        model=model,
    )

    return chat_completion.choices[0].message.content


if __name__ == '__main__':
    result = groq_inference(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        prompt="Hello. Poem about the meaning of life in 2 verses."
    )
    print(result)
