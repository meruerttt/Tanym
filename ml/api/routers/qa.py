from fastapi import APIRouter, HTTPException


from ml.api.schemas.qa import RAGResponse, RAGRequest, SourceDocument
from ml.model.rag_model.rag_inference import RAGPipeline
from ml.utils.logs.logging_config import logger


router = APIRouter(prefix="/rag", tags=["RAG QA"])

pipeline = RAGPipeline(
    embedding_model="sentence-transformers/all-mpnet-base-v2",
    llm_model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
    prompt_template_path="C:/Users/User/Tanym/ml/prompts/templates/grade_subject_prompt/qa.j2"
)

@router.post("/inference", response_model=RAGResponse)
def rag_endpoint(request: RAGRequest):
    try:
        logger.info(f"Запрос получен: grade={request.grade}, subject={request.subject}, question={request.question}")
        answer, docs = pipeline.run(request.subject, request.grade, request.question)
        sources = [
            SourceDocument(
                content=doc.page_content,
                page=str(doc.metadata.get("page")) if doc.metadata.get("page") else None
            )
            for doc in docs
        ]
        return RAGResponse(answer=answer, sources=sources)
    except FileNotFoundError as e:
        logger.error(f"Файл не найден: {e}")
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.exception("Непредвиденная ошибка")
        raise HTTPException(status_code=500, detail=f"Ошибка выполнения: {e}")
#
# from ml.model.router_prompts import assistant_chain
#
# @router.post("/assistant")
# def handle_question(data: RAGRequest):
#     return {"answer": assistant_chain.invoke(data.dict())}

