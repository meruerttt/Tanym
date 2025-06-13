from fastapi import APIRouter, HTTPException

from ml.model.rag_model.rag_query import rag_inference
from ml.api.schemas.qa import RAGResponse, RAGRequest, SourceDocument

router = APIRouter(prefix="/rag", tags=["RAG QA"])


@router.post("/inference", response_model=RAGResponse)
def rag_endpoint(request: RAGRequest):
    try:
        answer, docs = rag_inference(request.subject, request.grade, request.question)
        sources = [
            SourceDocument(
                content=doc.page_content,
                page=str(doc.metadata.get("page")) if doc.metadata.get("page") else None
            )
            for doc in docs
        ]
        return RAGResponse(answer=answer, sources=sources)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка выполнения: {e}")
