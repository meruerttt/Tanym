import os
import logging

from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from together import Together
from jinja2 import Template

from ml.model.rag_model.rag_reranker import CosineReranker
from ml.utils.logs.logging_config import logger


load_dotenv()


EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
TOGETHER_MODEL = "mistralai/Mistral-7B-Instruct-v0.1"

def rag_inference(subject: str, grade: int, question: str):
    index_path = f"C:/Users/User/Tanym/ml/data/vectordb/{grade}/{subject}/vector_data"
    prompt_path = f"C:/Users/User/Tanym/ml/prompts/templates/grade_subject_prompt/{grade}/{subject}/prompt.j2"

    if not os.path.exists(index_path):
        raise FileNotFoundError(f"Vector DB not found: {index_path}")
    if not os.path.exists(prompt_path):
        raise FileNotFoundError(f"Prompt file not found: {prompt_path}")

    logger.info(f"Загрузка эмбеддингов и векторной базы из {index_path}")
    embedding = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    db = FAISS.load_local(index_path, embedding, allow_dangerous_deserialization=True)
    retriever = db.as_retriever(search_kwargs={"k": 10})
    reranker = CosineReranker(embedding_model_name=EMBEDDING_MODEL)
    client = Together(api_key=os.getenv("TOGETHER_API_KEY"))

    logger.info(f"Извлечение документов по запросу: {question}")
    initial_docs = retriever.get_relevant_documents(question)
    docs = reranker.rerank(question, initial_docs, top_k=3)

    context = "\n\n".join([doc.page_content for doc in docs])

    logger.info(f"Формирование prompt из шаблона {prompt_path}")
    with open(prompt_path, "r", encoding="utf-8") as f:
        template = Template(f.read())
    full_prompt = template.render(context=context, question=question)

    logger.info(f"Отправка запроса к Together.ai")
    response = client.chat.completions.create(
        model=TOGETHER_MODEL,
        messages=[{"role": "user", "content": full_prompt}],
        max_tokens=300,
        temperature=0.5,
        stream=False
    )

    answer = response.choices[0].message.content.strip()
    logger.info(f"Ответ получен.")

    return answer, docs
