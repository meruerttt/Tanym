import pytest
try:
    from ml.model.saiga_model import generate
    from ml.prompts.prompt_for_feedback import prompt_for_check_task
    MODEL_AVAILABLE = True
except ImportError:
    MODEL_AVAILABLE = False

@pytest.mark.skipif(not MODEL_AVAILABLE, reason="Модель недоступна")
def test_generate_output():
    prompt = prompt_for_check_task("Математика", 5, "2+2", "5")
    result = generate(prompt)
    assert isinstance(result, str)
    assert len(result.strip()) > 10
    assert any(word in result.lower() for word in ["правильный", "ошибка", "молодец"])

