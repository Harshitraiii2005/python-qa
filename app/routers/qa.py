"""Q&A endpoints."""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from app.core.rag import rag_pipeline

router = APIRouter()


class AskRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=1000,
                          example="How do I read a CSV file in Python using pandas?")


class Source(BaseModel):
    title:     str
    score:     int
    relevance: float
    so_id:     str


class AskResponse(BaseModel):
    question:   str
    answer:     str
    sources:    list[Source]
    latency_ms: int
    model:      str


@router.post("/ask", response_model=AskResponse)
async def ask(req: AskRequest):
    """
    Submit a Python programming question and receive a grounded answer
    with citations from Stack Overflow.
    """
    if not rag_pipeline.ready:
        raise HTTPException(503, "Service initialising — please retry in ~30 s.")
    result = await rag_pipeline.ask(req.question)
    return AskResponse(**result)
