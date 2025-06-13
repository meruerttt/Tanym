import numpy as np
from typing import List
from langchain.schema import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity

class CosineReranker:
    def __init__(self, embedding_model_name: str):
        self.embedding_model = HuggingFaceEmbeddings(model_name=embedding_model_name)

    def rerank(self, query: str, docs: List[Document], top_k: int = 3) -> List[Document]:
        query_emb = self.embedding_model.embed_query(query)
        doc_embeddings = [self.embedding_model.embed_query(doc.page_content) for doc in docs]
        scores = cosine_similarity([query_emb], doc_embeddings)[0]
        sorted_indices = np.argsort(scores)[::-1][:top_k]
        return [docs[i] for i in sorted_indices]
