import os
from dotenv import load_dotenv
from jinja2 import Template

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from together import Together

from ml.model.rag_model.rag_reranker import CosineReranker
from ml.utils.logs.logging_config import logger


class RAGPipeline:
    def __init__(
        self,
        embedding_model: str,
        llm_model: str,
        prompt_template_path: str,
        vector_db_path: str = "C:/Users/User/Tanym/ml/data/vectordb",
        api_key_env: str = "TOGETHER_API_KEY"
    ):
        load_dotenv()
        self.embedding_model_name = embedding_model
        self.llm_model = llm_model
        self.prompt_template_path = prompt_template_path
        self.vector_db_path = vector_db_path
        self.api_key = os.getenv(api_key_env)
        if not self.api_key:
            raise ValueError("API key not found in environment variables.")

        logger.info("Инициализация моделей")
        self.embedding = HuggingFaceEmbeddings(model_name=self.embedding_model_name)
        self.client = Together(api_key=self.api_key)
        self.reranker = CosineReranker(embedding_model_name=self.embedding_model_name)

    def run(self, subject: str, grade: int, question: str):
        if not os.path.exists(self.vector_db_path):
            raise FileNotFoundError(f"Vector DB not found: {self.vector_db_path}")
        if not os.path.exists(self.prompt_template_path):
            raise FileNotFoundError(f"Prompt file not found: {self.prompt_template_path}")

        logger.info(f"Загрузка векторной базы из {self.vector_db_path}")
        db = FAISS.load_local(self.vector_db_path, self.embedding, allow_dangerous_deserialization=True)

        retriever = db.as_retriever(
            search_kwargs={
                "k": 10,
                "filters": {
                    "grade": str(grade),
                    "subject": subject.lower()
                }
            }
        )

        logger.info(f"Извлечение документов по запросу: {question}")
        initial_docs = retriever.invoke(question)
        docs = self.reranker.rerank(question, initial_docs, top_k=3)
        context = "\n\n".join([doc.page_content for doc in docs])

        logger.info(f"Формирование prompt из шаблона {self.prompt_template_path}")
        with open(self.prompt_template_path, "r", encoding="utf-8") as f:
            template = Template(f.read())
        full_prompt = template.render(
            context=context,
            question=question,
            grade=grade,
            subject=subject
        )

        logger.info(f"Отправка запроса к Together.ai")
        response = self.client.chat.completions.create(
            model=self.llm_model,
            messages=[{"role": "user", "content": full_prompt}],
            max_tokens=200,
            temperature=0.5,
            stream=False
        )

        answer = response.choices[0].message.content.strip()
        logger.info("Ответ получен")

        return answer, docs
