# GraphRAG Discussion Intelligence

A multi-agent Hybrid GraphRAG system that answers questions about Reddit discussions. When you ask a question, it:
1. Checks if the RAG already has relevant data
2. If not → pulls relevant Reddit posts + threaded comments from Pullpush.io → ingests into graph + vector DBs
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
    │         Agent 2: Crawler (Pullpush.io) → Reddit posts + threaded comments
    │                    ▼
    │         Agent 3: Parser (Ollama llama3.2:3b) → entities/topics/sentiment
    │                    ▼
    │         Agent 4: Ingestion → ChromaDB + Neo4j (posts + comment trees)
    └─────────┬──────────┘
              ▼
    Agent 5: Hybrid retrieval (graph + vector, manual RRF) → LLM Answer
```

> Earlier versions also did a DuckDuckGo web search + Crawl4AI page scraping. That path was removed to keep the knowledge base **Reddit-only** — it avoids off-topic web noise, removes the flaky web-search dependency, and makes runs faster.

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
3. **Reddit data** — no credentials needed. Uses [Pullpush.io](https://api.pullpush.io) for historical Reddit posts/comments (free, no auth)

### Install

```bash
git clone <your-repo-url>
cd Reddit_Scraper
pip install -r requirements.txt
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

---

## How It Works (Detailed Walkthrough)

This section explains the whole system end to end: the moving parts, what each one does, why each tool was chosen, and what happens when a single question flows through the pipeline.

### 1. The core idea — a self-healing RAG

Most RAG systems fail when the knowledge base doesn't contain the answer. This system treats that as a normal case: if it lacks data for your question, it **goes and collects it on the fly** (fetch relevant Reddit posts + comments → parse → store), then answers. The next time a similar question is asked, the data is already there and it answers instantly. So the knowledge base grows itself as it's used.

Two databases hold that knowledge, and they play different roles:
- **Neo4j (graph)** — captures *relationships*: who posted what, which post replies to which, what topics/entities a post discusses, who is active in which community, and *when*.
- **ChromaDB (vectors)** — captures *meaning*: semantic embeddings of every post and comment so we can find content by similarity even when keywords don't match.

Answers are produced by **fusing both** retrieval styles (graph + vector), which is what "Hybrid GraphRAG" means.

### 2. Orchestration — a LangGraph state machine

The whole pipeline is a **LangGraph `StateGraph`** (`agents/orchestrator.py`). Every agent is a node; a shared `AgentState` dict (`agents/state.py`) is passed from node to node, each one adding its results.

```
query_understanding ──► (has_rag_data?) ──► answer            (data already exists → answer directly)
                              │
                              └──► crawl ─► parse ─► ingest ─► answer   (no data → build it, then answer)
```

The decision point is a **conditional edge** (`_route_after_query`): the Query agent sets a `has_rag_data` flag, and the graph routes accordingly. This is what makes it "self-healing" — the expensive collection path only runs when needed.

### 3. The five agents

| # | Agent | File | Model / Tool | Job |
|---|-------|------|--------------|-----|
| 1 | **Query Understanding** | `agents/query_agent.py` | Ollama `llama3.1:8b` | Extracts intent (topics, entities, time range, comparison windows) as JSON, then checks whether the RAG already has enough relevant data. |
| 2 | **Crawler** | `agents/crawler_agent.py` | Pullpush.io | Pulls historical Reddit posts + their threaded comments from Pullpush (free, no auth). |
| 3 | **Parser** | `agents/parser_agent.py` | Ollama `llama3.2:3b` | Uses the LLM to extract topics/entities/sentiment per post; assigns a time window; carries comment threading metadata. |
| 4 | **Ingestion** | `agents/ingestion_agent.py` | ChromaDB + Neo4j | Chunks + embeds everything into the vector store, and writes the post/comment graph into Neo4j. |
| 5 | **Answer** | `agents/answer_agent.py` | Ollama `llama3.1:8b` | Runs hybrid retrieval (graph + vector), fuses with RRF, and generates a cited answer. |

> `agents/search_agent.py` (DuckDuckGo URL discovery) still exists in the repo but is **no longer wired into the pipeline** — see the Reddit-only note in [Architecture](#architecture).

#### Agent 1 — Query Understanding (the "router")
- Prompts the smart LLM to return structured intent: `{topics, entities, time_range, is_temporal_comparison, compare_windows}`.
- A regex helper (`utils/temporal.parse_comparison_windows`) also detects when two time windows are being compared (e.g. "Q4 2024 vs Q1 2025").
- **Coverage check** (`_check_coverage`) decides `has_rag_data`:
  1. If ChromaDB holds fewer than 10 docs → **no data** (cold start).
  2. Else run a semantic search; if ≥3 hits are similar enough (distance < 0.8) → **has data**.
  3. Else check Neo4j for matching topics (case-insensitive); ≥3 matches → **has data**.
- That boolean drives the conditional routing above. (The `< 0.8` threshold and case-insensitive topic match were tuned so a repeat question correctly recognizes existing data instead of re-fetching.)

#### Agent 2 — Crawler (Reddit via Pullpush)
- **Pullpush.io** (free Reddit historical API, no auth) fetches top submissions across the tracked subreddits (`MachineLearning`, `LocalLLaMA`) for each time window, filtered by the query's topics.
- For each post it also fetches comments and rebuilds a **threaded comment tree** (see [Comment ingestion](#comment-ingestion)).
- Each post carries its real **Reddit submission id** (`reddit_id`), which becomes the stable graph node id so re-fetching the same post never creates duplicates.
- Pullpush calls use a small **retry/backoff** to ride out rate limits.

#### Agent 3 — Parser
- For every post, the fast LLM returns `{topics, entities, sentiment}` as JSON (robustly parsed by `utils/json_parser.extract_json`, which tolerates messy LLM output).
- Each doc is stamped with a **time window** via `utils/temporal.epoch_to_window`, and given a **stable id** (`reddit_<id>`, or a URL hash for the rare non-Reddit doc).
- The comment tree is walked recursively to attach threading metadata (parent id, depth) and time windows. Comments are stored **without** a per-comment sentiment LLM call (this was dropped for speed — it previously meant hundreds of LLM calls per run).

#### Agent 4 — Ingestion (writes to both stores)
- **Vectors:** `vector/chunker.py` splits long posts into ~600-char paragraph chunks (`chunk_post`) and turns each comment into a chunk (`chunk_comment`); all are embedded with `all-MiniLM-L6-v2` and stored in ChromaDB. Each chunk is written with a **stable id** (`<post_id>:<chunk_index>`, `c:<comment_id>`), so re-ingesting upserts instead of piling up duplicates.
- **Graph:** `_write_to_neo4j` MERGEs `Post`, `Source`, `User`, `Topic`, `Entity` nodes and their relationships; `_write_comment_tree` recursively writes `Comment` nodes with `COMMENTS_ON` / `REPLY_TO` / `AUTHORED_BY` / `ACTIVE_IN` edges.
- All writes are `MERGE`-based on stable ids, so re-ingesting the same content is idempotent (no duplicate nodes).

#### Agent 5 — Answer (hybrid retrieval + generation)
- **Graph retrieval** (`graph/graph_retriever.py`): keyword-matches the query against post titles/bodies in Neo4j, optionally filtered to a time window.
- **Vector retrieval** (`vector/vector_store.py`): semantic similarity search in ChromaDB, optionally filtered by `time_window` metadata.
- The two ranked lists are merged with **Reciprocal Rank Fusion** (see [Hybrid Retrieval](#hybrid-retrieval)).
- The smart LLM then writes a comprehensive answer **with source citations**.
- **Temporal comparison** queries take a different branch: retrieval runs once per window and both contexts are fed to a compare prompt that highlights what changed between the two periods.

### 4. Data flow for one query (concrete example)

> **Question:** "How has sentiment around open-source LLMs changed recently?"

1. **Query agent** extracts `topics=["open-source LLMs", "sentiment"]`, finds the vector store empty → `has_rag_data = False`.
2. Router sends it down the **collection path**.
3. **Crawler** → Pullpush pulls r/LocalLLaMA + r/MachineLearning posts and their top comments → threaded trees (each post keyed by its Reddit id).
4. **Parser** → LLM tags topics/entities/sentiment per post; assigns time windows; threads comments.
5. **Ingestion** → embeds chunks into ChromaDB; builds the post/comment graph in Neo4j (idempotent MERGE on stable ids).
6. **Answer agent** → graph + vector retrieval → RRF fusion → LLM writes a cited answer.
7. Ask a similar question later → step 1 finds the data → it jumps straight to **answer** (no crawling).

### 5. Why these tools

| Concern | Choice | Why |
|---------|--------|-----|
| LLM | **Ollama** (`llama3.1:8b` + `llama3.2:3b`) | Runs fully **local & free** — no API keys, no per-token cost. A larger model answers/understands; a small fast model does high-volume extraction. |
| Graph DB | **Neo4j Desktop** | Native graph traversal for relationship/temporal questions ("who is active where, when"). |
| Vector DB | **ChromaDB** | Local, persistent, zero-setup embeddings store with metadata filtering. |
| Embeddings | **`all-MiniLM-L6-v2`** (sentence-transformers) | Free, local, fast, strong quality for short discussion text. |
| Orchestration | **LangGraph** | Explicit state machine with conditional routing — perfect for "branch only if data is missing". |
| Reddit data | **Pullpush.io** | Free historical Reddit posts **and comments** with no OAuth. |
| API | **FastAPI** | Async, typed, auto-generated `/docs`. |

> Design note: every external dependency is **free and keyless** by design — the system runs entirely on a local machine with Ollama + Neo4j Desktop, which is why no paid APIs (OpenAI, Brave, Reddit OAuth) appear anywhere.

### 6. Project structure

```
agents/          # the pipeline agents + LangGraph orchestrator + shared state
  ├─ orchestrator.py    # builds & runs the LangGraph state machine
  ├─ state.py           # AgentState (the dict passed between agents)
  ├─ query_agent.py     # Agent 1: intent + RAG coverage check
  ├─ crawler_agent.py   # Agent 2: Pullpush Reddit posts + comment trees
  ├─ parser_agent.py    # Agent 3: LLM topic/entity/sentiment extraction
  ├─ ingestion_agent.py # Agent 4: write to ChromaDB + Neo4j
  ├─ answer_agent.py    # Agent 5: hybrid retrieval + RRF + answer
  └─ search_agent.py    # (legacy) DuckDuckGo URL discovery — no longer wired in
graph/           # Neo4j client, schema (constraints/indexes), graph retriever
vector/          # ChromaDB store + chunking logic
utils/           # config (env settings), time-window helpers, JSON parsing
api/             # FastAPI app + routes (query, ingest, graph, demo)
demo.py          # CLI that runs 4 representative queries end to end
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/query` | Run a query through the full multi-agent pipeline |
| `POST` | `/api/ingest` | Manually warm the RAG by ingesting Reddit data for a list of topics |
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
  "agents_used": ["query_agent", "crawler_agent", "parser_agent", "ingestion_agent", "answer_agent"],
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
- `(Comment)-[:COMMENTS_ON]->(Post)` — top-level comments
- `(Comment)-[:REPLY_TO]->(Comment)` — nested replies
- `(Comment)-[:AUTHORED_BY]->(User)`
- `(User)-[:ACTIVE_IN {time_window}]->(Source)`

**Time Windows**: `Q4_2024` (Oct–Dec 2024), `Q1_2025` (Jan–Mar 2025), `Q2_2025` (Apr–Jun 2025)

### Comment ingestion

Reddit comments are stored as first-class `Comment` nodes, threaded into a tree and selected by upvotes to keep the graph high-signal:

- **Top-level**: the top **5 comments per post** by score (fewer if the post has fewer).
- **Replies**: the top **3 replies per comment** by score, **fully nested** (applied recursively at every depth, capped at depth 8).
- Each comment is **embedded into ChromaDB** so comments participate in vector retrieval, and becomes a `Comment` node keyed by its **real Reddit comment id** (so replies thread correctly and re-ingestion never duplicates).
- Comments are stored **without** a per-comment sentiment LLM call. This was intentionally dropped: scoring every comment individually meant hundreds of LLM calls per run and made ingestion very slow. (Posts still get full topic/entity/sentiment extraction.)

Thresholds live as constants in `agents/crawler_agent.py` (`COMMENT_TOP_N`, `REPLY_TOP_N`, `COMMENT_MAX_DEPTH`) — lower them if ingestion feels slow on comment-heavy threads.

## Hybrid Retrieval

Graph traversal (Neo4j) and semantic search (ChromaDB) are run independently, then fused with a manual **Reciprocal Rank Fusion** pass (`reciprocal_rank_fusion` in `agents/answer_agent.py`):

```
RRF_score(doc) = Σ  1 / (k + rank_in_list)      # k = 60
```

This ensures documents that rank well in both graph traversal AND semantic search score highest. Since comments are embedded alongside posts, both posts and high-signal comments can surface in the fused results.
