import json
import os
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

LOG_PATH = os.getenv("LOG_PATH_FEEDBACK")

def log_feedback_entry(subject, grade, task, student_answer, feedback, model_version, template_version, human_rating=None):
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "subject": subject,
        "grade": grade,
        "task": task,
        "student_answer": student_answer,
        "generated_feedback": feedback,
        "model_version": model_version,
        "template_version": template_version,
        "human_rating": human_rating
    }
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
