import os

from dotenv import load_dotenv
from jinja2 import Environment, FileSystemLoader

load_dotenv()

env = Environment(loader=FileSystemLoader(os.getenv("PROMPT_PATH_FOR_FEEDBACK")))

def prompt_for_check_task(subject: str, grade: int, task: str, student_answer: str) -> str:
    template = env.get_template("feedback_prompt.j2")
    return template.render(
        subject=subject,
        grade=grade,
        task=task,
        student_answer=student_answer
    )
