from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from agents.orchestrator import run_pipeline

router = APIRouter()


class QueryRequest(BaseModel):
    query: str


class QueryResponse(BaseModel):
    query: str
    answer: str
    agents_used: list[str]
    graph_results: list[dict]
    vector_results: list[dict]
    sources: list[dict]
    has_rag_data: bool
    ingested_new_docs: bool


@router.post("/query", response_model=QueryResponse)
def query_endpoint(req: QueryRequest):
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    try:
        state = run_pipeline(req.query)
        return QueryResponse(
            query=req.query,
            answer=state.get("answer", ""),
            agents_used=state.get("agents_used", []),
            graph_results=state.get("graph_results", []),
            vector_results=state.get("vector_results", []),
            sources=state.get("sources", []),
            has_rag_data=state.get("has_rag_data", False),
            ingested_new_docs=state.get("ingested", False),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
