from fastapi import APIRouter
from agents.orchestrator import run_pipeline

router = APIRouter()

DEMO_QUERIES = [
    {
        "type": "semantic",
        "description": "Vector-dominant: semantic similarity over discussion content",
        "query": "What technical challenges do people face when fine-tuning LLMs locally?",
    },
    {
        "type": "graph_traversal",
        "description": "Graph-dominant: relationship and influence traversal",
        "query": "Who are the most active voices discussing AI safety and what communities are they in?",
    },
    {
        "type": "hybrid",
        "description": "Hybrid: needs both semantic context and graph relationships",
        "query": "How has community sentiment around open-source AI models evolved?",
    },
    {
        "type": "temporal_comparison",
        "description": "Time-series: comparing discussions across two time windows",
        "query": "Compare AI discussions in Q4 2024 vs Q1 2025 — what topics emerged or faded?",
    },
]


@router.get("/demo")
def demo_endpoint():
    results = []
    for demo in DEMO_QUERIES:
        print(f"\n{'='*60}")
        print(f"[Demo] Running: {demo['type']}")
        print(f"[Demo] Query: {demo['query']}")
        state = run_pipeline(demo["query"])
        results.append({
            "type": demo["type"],
            "description": demo["description"],
            "query": demo["query"],
            "answer": state.get("answer", ""),
            "agents_used": state.get("agents_used", []),
            "graph_results_count": len(state.get("graph_results", [])),
            "vector_results_count": len(state.get("vector_results", [])),
            "sources": state.get("sources", []),
            "ingested_new": state.get("ingested", False),
        })
    return {"demo_results": results}
