from together import Together
import os
from dotenv import load_dotenv

load_dotenv()

client = Together(api_key=os.getenv("TOGETHER_API_KEY"))


def generate(prompts: str):
    responce = client.chat.completions.create(
        model="mistralai/Mistral-7B-Instruct-v0.1",
        messages=[{
            "role": "user",
            "content": prompts
        }],
        max_tokens=300,
        temperature=0.5,
        stream=False
    )
    return responce.choices[0].message.content