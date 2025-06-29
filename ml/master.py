import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from ml.models.model import generate as generate_response

app = FastAPI(title="Tanym!")

class QuestionRequest(BaseModel):
    question: str = Field(..., max_length=1500, description="Вопрос пользователя (до 1500 символов)!!!")
    session_id: str = Field(default="default", description="Идентификатор сессии пользователя")

@app.post("/generate")
async def generate(request: QuestionRequest):
    try:
        response = await generate_response(request.question, request.session_id)
        return {"answer": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app=app, host="127.0.0.1", port=8888)