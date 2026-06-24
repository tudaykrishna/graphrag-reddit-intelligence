from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from agents.crawler_agent import run_crawler_agent
from agents.parser_agent import run_parser_agent
from agents.ingestion_agent import run_ingestion_agent

router = APIRouter()


class IngestRequest(BaseModel):
    urls: list[str]


class IngestResponse(BaseModel):
    urls_crawled: int
    docs_parsed: int
    chunks_stored: int
    status: str


@router.post("/ingest", response_model=IngestResponse)
def ingest_endpoint(req: IngestRequest):
    if not req.urls:
        raise HTTPException(status_code=400, detail="No URLs provided")

    try:
        state: dict = {"query": "", "search_urls": req.urls, "agents_used": []}
        state = run_crawler_agent(state)
        state = run_parser_agent(state)
        parsed = state.get("parsed_docs", [])
        state = run_ingestion_agent(state)

        return IngestResponse(
            urls_crawled=len(state.get("crawled_content", [])),
            docs_parsed=len(parsed),
            chunks_stored=len(parsed) * 2,
            status="success",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
