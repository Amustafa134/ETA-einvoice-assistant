"""
FastAPI app exposing the RAG pipeline as a real, askable service.

Run with:
    uvicorn src.api.app:app --reload --port 8000

Then open http://localhost:8000/docs for an interactive UI to ask
questions directly -- no separate frontend needed for a portfolio demo.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.generation.rag_pipeline import answer_question

app = FastAPI(
    title="ETA E-Invoice Assistant",
    description="Bilingual (Arabic/English) RAG assistant over Egypt's official e-invoicing regulations.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class QuestionRequest(BaseModel):
    question: str = Field(..., min_length=3, description="Ask in Arabic or English")
    top_k: int = Field(default=5, ge=1, le=20, description="How many chunks to retrieve")


class Source(BaseModel):
    source_file: str
    page_number: int
    score: float
    text_preview: str


class AnswerResponse(BaseModel):
    question: str
    answer: str
    sources: list[Source]


@app.get("/")
def root():
    return {
        "status": "ok",
        "service": "ETA E-Invoice Assistant",
        "ask_endpoint": "/ask (POST)",
        "docs": "/docs",
    }


@app.get("/health")
def health():
    """Basic health check -- confirms the API itself is up (not Qdrant/Azure)."""
    return {"status": "healthy"}


@app.post("/ask", response_model=AnswerResponse)
def ask(request: QuestionRequest):
    """
    Ask a question in Arabic or English about Egypt's e-invoicing system.
    Returns a grounded answer plus the real source chunks it was based on,
    with page-level citations.
    """
    try:
        result = answer_question(request.question, top_k=request.top_k)
    except Exception as e:
        print(f"Error answering question: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate an answer. Please try again.")

    sources = [
        Source(
            source_file=s["source_file"],
            page_number=s["page_number"],
            score=s["score"],
            text_preview=s["text"][:150],
        )
        for s in result["sources"]
    ]

    return AnswerResponse(
        question=result["question"],
        answer=result["answer"],
        sources=sources,
    )
