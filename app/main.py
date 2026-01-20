from fastapi import FastAPI
from app.api.v1 import router as api_v1
from app.core.config import settings

app = FastAPI(
    title=settings.PROJECT_NAME,
    version="1.0.0",
    description="Insight RAG API using LangChain 0.3 + Chroma"
)

app.include_router(api_v1, prefix=settings.API_V1_STR)

@app.get("/health", tags=["health"])
async def health():
    return {"status": "ok"}

@app.get("/")
async def root():
    return {"message": "Insight RAG API is running"}
