import pytest
from ml.prompts.promt_for_feedback import prompt_for_check_task

@pytest.fixture
def example_prompt():
    return prompt_for_check_task(
        subject="Математика",
        grade=5,
        task="Реши 2 + 2",
        student_answer="5"
    )

def test_prompt(example_prompt):
    assert isinstance(example_prompt, str)
    assert "Вы — учитель предмета" in example_prompt
    assert "2 + 2" in example_prompt or "2+2" in example_prompt
    assert "5" in example_prompt

# def test_prompt_template_is_stable(example_prompt):
#     hash_prompt = hash(example_prompt)
#     known_hash = -13578689452345412
#     assert hash_prompt == known_hash, "Prompt template изменился, проверь шаблон."