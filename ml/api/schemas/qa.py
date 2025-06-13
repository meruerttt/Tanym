from typing import List
from pydantic import BaseModel


class RAGRequest(BaseModel):
    subject: str
    grade: int
    question: str


class SourceDocument(BaseModel):
    content: str
    page: str | None = None


class RAGResponse(BaseModel):
    answer: str
    sources: List[SourceDocument]
