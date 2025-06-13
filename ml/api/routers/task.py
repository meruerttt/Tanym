from fastapi import FastAPI, APIRouter, status, HTTPException

from ml.utils.logs.logging_config import logger
from ml.api.schemas.task import TaskResponse, TaskRequest
from ml.model.saiga_model import generate
from ml.prompts.promt_for_feedback import prompt_for_check_task
from ml.utils.logs.logger import log_feedback_entry

router = APIRouter(prefix="/check_task", tags=["Check Task"])

@router.post(
    "/check_task",
    summary="Проверить задание ученика",
    response_model=TaskResponse,
    status_code=status.HTTP_200_OK
)
async def check_task(req: TaskRequest):
    try:
        logger.info(f"Обрабатываем задание по {req.subject} для {req.grade} класса")
        prompt = prompt_for_check_task(
            subject=req.subject,
            grade=req.grade,
            task=req.task,
            student_answer=req.student_answer
        )
        result = generate(prompt)
        if not result:
            logger.error("Модель не дала фидбэк")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Не удалось сгенерировать фидбэк, что-то пошло не так"
            )
        logger.info("Фидбэк готов, всё ок!")

        log_feedback_entry(
            subject=req.subject,
            grade=req.grade,
            task=req.task,
            student_answer=req.student_answer,
            feedback=result,
            model_version="saiga-mistral-q4",
            template_version="feedback_prompt_v1.1"
        )

        return TaskResponse(feedback=result)
    except Exception as e:
        logger.error(f"Ошибка при обработке: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ой, что-то сломалось: {str(e)}"
        )