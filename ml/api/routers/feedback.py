from fastapi import HTTPException, APIRouter, status

from ml.utils.logs.logging_config import logger
from ml.api.schemas.feedback import FeedbackResponse

router = APIRouter(prefix="/feedback", tags=["FEEDBACK"])

@router.get(
    "/feedback",
    summary="Получить оценку задания",
    response_model=FeedbackResponse,
    status_code=status.HTTP_200_OK
)
async def feedback(assessment: str):
    if assessment not in ["good", "bad"]:
        logger.warning(f"Неправильная оценка: {assessment}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Оценка должна быть либо 'good', либо 'bad', а не что попало!"
        )
    logger.info(f"Возвращаем оценку: {assessment}")
    return FeedbackResponse(assessment=assessment)
