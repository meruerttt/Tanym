import ctypes
import os
from llama_cpp import Llama, llama_log_set

from ml.prompts.promt_for_feedback import prompt_for_check_task




LOG_CALLBACK = ctypes.CFUNCTYPE(None, ctypes.c_int, ctypes.c_char_p)

@LOG_CALLBACK
def silent_logger(level: int, message: bytes):
    pass

llama_log_set(silent_logger, None)




model_path = os.path.abspath("C:/Users/User/Tanym/ml/save_models/Saiga-MIstral-7b/model-q4_K.gguf")
try:
    llm = Llama(
        model_path=model_path,
        n_ctx=4096,
        n_threads=6,
        use_mlock=False,
        use_mmap=True,
        verbose=False
    )
except Exception as e:
    print(f"Не удалось загрузить модель: {e}")
    exit()




def generate(prompt: str):
    response = llm(
        prompt,
        max_tokens=800,
        temperature=0.3,
        top_p=0.95,
        repeat_penalty=1.1,
        stop=["---", "Задание:", "Следующее задание"],
        echo=False,
        stream=False
    )
    return response["choices"][0]["text"].strip()


