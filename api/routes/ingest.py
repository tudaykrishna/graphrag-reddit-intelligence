from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from agents.crawler_agent import run_crawler_agent
from agents.parser_agent import run_parser_agent
from agents.ingestion_agent import run_ingestion_agent

router = APIRouter()


class IngestRequest(BaseModel):
    topics: list[str]


class IngestResponse(BaseModel):
    docs_crawled: int
    docs_parsed: int
    status: str


@router.post("/ingest", response_model=IngestResponse)
def ingest_endpoint(req: IngestRequest):
    """Manually warm the RAG by ingesting Reddit data for the given topics."""
    if not req.topics:
        raise HTTPException(status_code=400, detail="No topics provided")

    try:
        state: dict = {"query": " ".join(req.topics), "intent": {"topics": req.topics}, "agents_used": []}
        state = run_crawler_agent(state)
        state = run_parser_agent(state)
        parsed = state.get("parsed_docs", [])
        run_ingestion_agent(state)

        return IngestResponse(
            docs_crawled=len(state.get("crawled_content", [])),
            docs_parsed=len(parsed),
            status="success",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
