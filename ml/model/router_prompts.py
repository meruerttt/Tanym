import os
from dotenv import load_dotenv
from langchain_core.runnables import Runnable
from jinja2 import Template

from ml.model.rag_model.rag_inference import RAGPipeline
from ml.model.local_model.check import generate

load_dotenv()

TEMPLATE_DIR = "C:/Users/User/Tanym/ml/prompts/templates/grade_subject_prompt"
TASK_TYPES = ["explanation", "feedback", "qa", "search"]


def get_jinja_prompt(task_type: str) -> str:
    path = os.path.join(TEMPLATE_DIR, f"{task_type}.j2")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Не найден шаблон {task_type}")
    with open(path, encoding="utf-8") as f:
        return f.read()


def classify_mode(question: str) -> str:
    """RAG или Prompt-only?"""
    prompt = f"""
Ты ассистент, который определяет, как ответить на вопрос: через базу знаний (RAG) или по шаблону (prompt-only).
Ответь одним словом: rag или prompt.

Вопрос: {question}
"""
    result = generate(prompt)
    return "rag" if "rag" in result.lower() else "prompt"


def classify_task_type(question: str) -> str:
    """Определяет тип задачи"""
    prompt = f"""
Ты классификатор. Прочитай вопрос и определи, к какому типу задач он относится:
- explanation (если нужно объяснить тему),
- feedback (если нужно проверить задание),
- qa (если это вопрос из теста),
- search (если нужно найти материалы).

Ответь одним словом.

Вопрос: {question}
"""
    result = generate(prompt)
    for task in TASK_TYPES:
        if task in result.lower():
            return task
    return "explanation"  # fallback


class RouterAssistant(Runnable):
    def invoke(self, input: dict) -> str:
        subject = input["subject"]
        grade = input["grade"]
        question = input["question"]

        task_type = classify_task_type(question)
        mode = classify_mode(question)

        if mode == "rag":
            rag = RAGPipeline(
                embedding_model="sentence-transformers/all-mpnet-base-v2",
                llm_model="mistralai/Mistral-7B-Instruct-v0.1",
                prompt_template_path=os.path.join(TEMPLATE_DIR, f"{task_type}.j2"),
            )
            return rag.run(subject, grade, question)[0]

        else:
            template = Template(get_jinja_prompt(task_type))
            prompt = template.render(subject=subject, grade=grade, question=question, context="")
            return generate(prompt)


assistant_chain = RouterAssistant()
