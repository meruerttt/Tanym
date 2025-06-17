import uvicorn
from fastapi import FastAPI

from ml.api.routers import qa, feedback


app = FastAPI(
    title="Таным для Образования",
    description="API для проверки заданий учеников",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

app.include_router(qa.router)
app.include_router(feedback.router)

@app.get("/", summary="Главная страница", response_model=dict)
async def root():
    return {"message": "API Таным для образования запущен"}

if __name__ == "__main__":
    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8002,
        log_level="info"
    )