from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.routes import query, ingest, graph, demo

app = FastAPI(
    title="GraphRAG Discussion Intelligence",
    description="Multi-agent hybrid GraphRAG system for discussion intelligence",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(query.router, prefix="/api", tags=["Query"])
app.include_router(ingest.router, prefix="/api", tags=["Ingest"])
app.include_router(graph.router, prefix="/api", tags=["Graph"])
app.include_router(demo.router, prefix="/api", tags=["Demo"])


@app.get("/")
def root():
    return {
        "message": "GraphRAG Discussion Intelligence API",
        "docs": "/docs",
        "endpoints": {
            "query": "POST /api/query",
            "ingest": "POST /api/ingest",
            "demo": "GET /api/demo",
            "topics": "GET /api/graph/topics",
            "users": "GET /api/graph/users",
            "stats": "GET /api/stats",
        },
    }
