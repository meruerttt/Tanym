# test_logger.py
import os
import json
from unittest.mock import patch
from ml.utils.logs import logger

def test_log_feedback_entry(tmp_path):
    log_file = tmp_path / "feedback_test_log.jsonl"

    with patch.dict(os.environ, {"LOG_PATH_FEEDBACK": str(log_file)}):
        logger.LOG_PATH = str(log_file)
        logger.log_feedback_entry(
            subject="Биология",
            grade=10,
            task="Сколько хромосом у человека?",
            student_answer="47",
            feedback="Ты был близок, но правильный ответ - 46",
            model_version="saiga-test",
            template_version="v1.0"
        )

        assert log_file.exists()
        with open(log_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
            assert len(lines) == 1
            data = json.loads(lines[0])
            assert data["subject"] == "Биология"
            assert "правильный ответ" in data["generated_feedback"]
            assert data["model_version"] == "saiga-test"
