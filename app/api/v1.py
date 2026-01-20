from fastapi import APIRouter, HTTPException
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel
from app.db.vector_store import VectorStoreManager
from app.core.rag_service import RAGService
from app.core.document_processor import SeniorSemanticProcessor
from app.core.config import settings

router = APIRouter()


class QueryRequest(BaseModel):
    question: str


@router.post("/ingest")
async def ingest_data():
    try:
        processor = SeniorSemanticProcessor()

        docs = await run_in_threadpool(
            processor.process_metadata,
            settings.METADATA_PATH
        )

        v_manager = VectorStoreManager()
        await run_in_threadpool(
            v_manager.initialize_db,
            docs
        )

        return {
            "status": "success",
            "method": "semantic_chunking",
            "chunks_created": len(docs),
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/ask")
async def ask_question(request: QueryRequest):
    try:
        v_manager = VectorStoreManager()
        retriever = v_manager.get_retriever()

        docs = await run_in_threadpool(
            retriever.get_relevant_documents,
            "health_check"
        )
        if not docs:
            raise HTTPException(
                status_code=400,
                detail="Database not initialized. Please call /ingest first.",
            )

        rag = RAGService(retriever)

        answer = await run_in_threadpool(
            rag.answer_query,
            request.question
        )

        return {"answer": answer}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
