# GraphRAG Discussion Intelligence

A multi-agent Hybrid GraphRAG system that answers questions about online discussions. When you ask a question, it:
1. Checks if the RAG already has relevant data
2. If not → searches Brave API for discussion URLs → crawls them → ingests into graph + vector DBs
3. Answers using hybrid retrieval (graph traversal + vector search, fused with RRF)

## Architecture

```
User Query
    │
    ▼
Agent 1: Query Understanding → checks Neo4j + ChromaDB coverage
    │                    │
  has data           no data
    │                    ▼
    │         Agent 2: Brave Search → URLs
    │                    ▼
    │         Agent 3: Crawl4AI + Pullpush.io → raw content
    │                    ▼
    │         Agent 4: Parser (Ollama llama3.2:3b) → entities/topics/sentiment
    │                    ▼
    │         Agent 5: Ingestion → ChromaDB + Neo4j
    └─────────┬──────────┘
              ▼
    Agent 6: EnsembleRetriever (graph + vector, RRF) → LLM Answer
```

**Orchestration**: LangGraph state machine with conditional routing  
**Graph DB**: Neo4j Desktop (temporal nodes + relationships)  
**Vector DB**: ChromaDB (local persistent, metadata-filtered)  
**LLM**: Ollama `llama3.1:8b` (answers) + `llama3.2:3b` (extraction)  
**Embeddings**: `sentence-transformers/all-MiniLM-L6-v2` (free, local)

## Setup

### Prerequisites

1. **Neo4j Desktop** — start your local DBMS, note the password
2. **Ollama** — install from [ollama.ai](https://ollama.ai) and pull models:
   ```bash
   ollama pull llama3.1:8b
   ollama pull llama3.2:3b
   ollama serve
   ```
3. **Brave Search API** — free account at [search.brave.com/search/api](https://search.brave.com/search/api) (2000 searches/month free)
4. **Reddit data** — no credentials needed. Uses:
   - [Pullpush.io](https://api.pullpush.io) for historical Reddit posts/comments (free, no auth)
   - [Crawl4AI](https://github.com/unclecode/crawl4ai) for live Reddit page scraping

### Install

```bash
git clone <your-repo-url>
cd Reddit_Scraper
pip install -r requirements.txt
playwright install chromium   # for Crawl4AI JS rendering
```

### Configure

```bash
cp .env.example .env
# Edit .env and fill in all values
```

### Run

```bash
# Start the API server
uvicorn api.main:app --reload --port 8000

# Or run the CLI demo directly
python demo.py
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/query` | Run a query through the full multi-agent pipeline |
| `POST` | `/api/ingest` | Manually ingest content from a list of URLs |
| `GET` | `/api/demo` | Run all 4 demo queries and return results |
| `GET` | `/api/graph/topics` | List topics in Neo4j (optional `?time_window=Q1_2025`) |
| `GET` | `/api/graph/users` | Top influential users (optional `?time_window=Q4_2024`) |
| `GET` | `/api/graph/topic/{name}/evolution` | Track topic mentions across time windows |
| `GET` | `/api/stats` | Document counts |

Interactive docs: [http://localhost:8000/docs](http://localhost:8000/docs)

### Example Query

```bash
curl -X POST http://localhost:8000/api/query \
  -H "Content-Type: application/json" \
  -d '{"query": "How has sentiment around open-source LLMs changed in the last 6 months?"}'
```

Response:
```json
{
  "query": "...",
  "answer": "Based on discussions across r/LocalLLaMA and r/MachineLearning...",
  "agents_used": ["query_agent", "search_agent", "crawler_agent", "parser_agent", "ingestion_agent", "answer_agent"],
  "graph_results": [...],
  "vector_results": [...],
  "sources": [{"url": "...", "time_window": "Q1_2025"}],
  "has_rag_data": false,
  "ingested_new_docs": true
}
```

## Knowledge Graph Schema

**Nodes**: `Post`, `Comment`, `User`, `Source`, `Topic`, `Entity`  
**Relationships** (all timestamped):
- `(Post)-[:FROM_SOURCE]->(Source)`
- `(Post)-[:AUTHORED_BY]->(User)`
- `(Post)-[:DISCUSSES {time_window}]->(Topic)`
- `(Post)-[:MENTIONS {time_window}]->(Entity)`
- `(User)-[:ACTIVE_IN {time_window}]->(Source)`

**Time Windows**: `Q4_2024` (Oct–Dec 2024), `Q1_2025` (Jan–Mar 2025), `Q2_2025` (Apr–Jun 2025)

## Hybrid Retrieval

Uses LangChain `EnsembleRetriever` with weights `[0.5, 0.5]` which implements **Reciprocal Rank Fusion**:

```
RRF_score(doc) = Σ  1 / (k + rank_in_list)
```

This ensures documents that rank well in both graph traversal AND semantic search score highest.
