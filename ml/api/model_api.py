import uvicorn
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field
import logging


from ml.model.saiga_model import generate
from ml.prompts.promt_for_feedback import prompt_for_check_task



logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



app = FastAPI(
    title="Таным для Образования",
    description="API для проверки заданий учеников и дачи фидбэка",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)




class TaskRequest(BaseModel):
    subject: str = Field(..., min_length=1, description="Предмет, типа математика или литература")
    grade: int = Field(..., ge=1, le=11, description="Класс ученика, от 1 до 11")
    task: str = Field(..., min_length=1, description="Само задание, что задали ученику")
    student_answer: str = Field(..., min_length=1, description="Ответ ученика, что он там написал")

    class Config:
        json_schema_extra = {
            "example": {
                "subject": "Математика",
                "grade": 5,
                "task": "Реши уравнение: 2x + 3 = 11",
                "student_answer": "x = 4"
            }
        }

class TaskResponse(BaseModel):
    feedback: str = Field(..., description="Фидбэк по ответу ученика, с оценкой и советами")

class FeedbackRequest(BaseModel):
    assessment: str = Field(..., pattern="^(good|bad)$", description="Оценка задания")

class FeedbackResponse(BaseModel):
    assessment: str = Field(..., description="То, что передали в оценке")




@app.get("/", summary="Главная страница", response_model=dict)
async def root():
    return {"message": "API Таным для образования запущен"}



@app.post(
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
        return TaskResponse(feedback=result)
    except Exception as e:
        logger.error(f"Ошибка при обработке: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ой, что-то сломалось: {str(e)}"
        )





@app.get(
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



if __name__ == "__main__":
    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8002,
        log_level="info"
    )