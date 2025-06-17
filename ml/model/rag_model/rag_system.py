import os

from typing import List
from dataclasses import dataclass
from dotenv import load_dotenv

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain.schema import Document

from ml.utils.logs.logging_config import logger


load_dotenv()


@dataclass
class RAGConfig:
    folder_path: str = "C:/Users/User/Tanym/ml/data/subject_grade_data"
    vector_store_path: str = "C:/Users/User/Tanym/ml/data/vectordb"
    embedding_model: str = "sentence-transformers/all-mpnet-base-v2"
    chunk_size: int = 1000
    chunk_overlap: int = 150


def extract_metadata_from_path(path: str) -> dict:
    parts = os.path.normpath(path).split(os.sep)
    try:
        grade = parts[-3]
        subject = parts[-2]
    except IndexError:
        grade = "unknown"
        subject = "unknown"
    return {"grade": grade, "subject": subject, "filename": os.path.basename(path)}


def load_all_pdfs(config: RAGConfig) -> List[Document]:
    logger.info("Загрузка всех PDF из папки")
    all_docs = []
    for root, _, files in os.walk(config.folder_path):
        for file in files:
            if file.endswith(".pdf"):
                pdf_path = os.path.join(root, file)
                try:
                    loader = PyPDFLoader(pdf_path)
                    docs = loader.load()
                    metadata = extract_metadata_from_path(pdf_path)
                    for doc in docs:
                        doc.metadata.update(metadata)
                    all_docs.extend(docs)
                except Exception as e:
                    logger.error(f"Ошибка при загрузке {pdf_path}: {e}")
    return all_docs


def split_docs(docs: List[Document], config: RAGConfig) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap
    )
    return splitter.split_documents(docs)


def build_faiss_index(docs: List[Document], config: RAGConfig):
    logger.info("Создание FAISS индекса")
    embeddings = HuggingFaceEmbeddings(model_name=config.embedding_model)
    db = FAISS.from_documents(docs, embedding=embeddings)
    db.save_local(config.vector_store_path)
    logger.info("Индекс сохранён в %s", config.vector_store_path)


if __name__ == "__main__":
    config = RAGConfig()
    docs = load_all_pdfs(config)
    chunks = split_docs(docs, config)
    build_faiss_index(chunks, config)
