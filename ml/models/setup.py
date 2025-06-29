from langchain_together import Together
from langchain.prompts import PromptTemplate
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory

from ml.models import config
import os

os.environ["TOGETHER_API_KEY"] = config.TOGETHER_API_KEY

with open(config.PROMPT_TEMPLATE_PATH, "r", encoding="utf-8") as f:
    template_str = f.read()

prompt = PromptTemplate(
    input_variables=["input", "chat_history"],
    template=template_str
)

llm = Together(
    model=config.LLM_MODEL_NAME,
    temperature=config.TEMPERATURE,
    max_tokens=config.MAX_TOKENS,
    top_p=config.TOP_P,
    repetition_penalty=config.REPETITION_PENALTY,
)

chain = prompt | llm

def get_session_history(session_id: str):
    return InMemoryChatMessageHistory()

chain_with_memory = RunnableWithMessageHistory(
    chain,
    get_session_history=get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
)
