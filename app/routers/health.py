"""Health check endpoints."""
from fastapi import APIRouter
from app.core.rag import rag_pipeline

router = APIRouter()


@router.get("/health")
async def health():
    """Returns service health and pipeline stats."""
    stats = rag_pipeline.stats()
    return {
        "status":  "ok" if rag_pipeline.ready else "initialising",
        "service": "python-qa-assistant",
        "version": "1.0.0",
        **stats,
    }
