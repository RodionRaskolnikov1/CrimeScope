from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter()


class ChatRequest(BaseModel):
    query: str


@router.post("/ask")
async def ask_question(req: ChatRequest):
    """RAG-powered Q&A over crime data."""
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
    try:
        from crimescope.nlp.qa_chain import ask
        result = ask(req.query)
        return {
            "query": result["query"],
            "answer": result["answer"],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))