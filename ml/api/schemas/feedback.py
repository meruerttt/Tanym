from pydantic import Field, BaseModel

class FeedbackRequest(BaseModel):
    assessment: str = Field(..., pattern="^(good|bad)$", description="Оценка задания")

class FeedbackResponse(BaseModel):
    assessment: str = Field(..., description="То, что передали в оценке")
