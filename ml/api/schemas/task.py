from pydantic import BaseModel, Field


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