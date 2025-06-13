from together import Together
from ml.prompts.promt_for_feedback import prompt_for_check_task
import os
from dotenv import load_dotenv

load_dotenv()


client = Together(api_key=os.getenv("TOGETHER_API_KEY"))

prompt = prompt_for_check_task("Биология", 11, "Сколько у человека хромосом?", "47")

responce = client.chat.completions.create(
    model="mistralai/Mistral-7B-Instruct-v0.1",
    messages=[{
        "role": "user",
        "content": prompt
    }],
    max_tokens=300,
    temperature=0.5,
    stream=False
)

print(responce.choices[0].message.content)