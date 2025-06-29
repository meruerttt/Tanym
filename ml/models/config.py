import os
from dotenv import load_dotenv

load_dotenv()

TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")
if not TOGETHER_API_KEY:
    raise ValueError("TOGETHER_API_KEY не найден в .env")

PROMPT_TEMPLATE_PATH = os.path.join(
    "C:/Users/User/Tanym/ml/utils/prompts/templates", "prompt.j2"
)

LLM_MODEL_NAME = "deepseek-ai/DeepSeek-V3"
TEMPERATURE = 0.4
MAX_TOKENS = 512
TOP_P = 0.9
REPETITION_PENALTY = 1.15

MEMORY_K = 6
