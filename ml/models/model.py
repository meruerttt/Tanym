import asyncio
from ml.models.setup import chain_with_memory

async def generate(question: str, session_id: str = "default") -> str:
    try:
        response = await asyncio.to_thread(
            chain_with_memory.invoke,
            input={"input": question},
            config={"configurable": {"session_id": session_id}},
        )
        return response
    except Exception as e:
        logger.exception("Ошибка при генерации ответа")
        return "Произошла ошибка при генерации."
