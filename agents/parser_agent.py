"""Agent 4 — Parse raw crawled content, extract entities/topics/sentiment via LLM."""
import hashlib
import time
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from agents.state import AgentState
from utils.config import get_settings
from utils.json_parser import extract_json
from utils.temporal import epoch_to_window

EXTRACT_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "You extract structured metadata from discussion content. Return only valid JSON."),
    ("human", """Analyze this discussion content and return ONLY a JSON object:

Content (first 800 chars):
{content}

Return this exact structure:
{{
  "topics": ["topic1", "topic2", "topic3"],
  "entities": [
    {{"name": "GPT-4", "type": "MODEL"}},
    {{"name": "OpenAI", "type": "COMPANY"}}
  ],
  "sentiment": {{"label": "positive", "score": 0.7}}
}}

Entity types: MODEL, COMPANY, CONCEPT, PERSON, TOOL
Sentiment: positive / negative / neutral, score 0.0-1.0
Topics: 2-4 short theme phrases

Return ONLY the JSON, no explanation."""),
])


def run_parser_agent(state: AgentState) -> AgentState:
    raw_docs = state.get("crawled_content", [])
    if not raw_docs:
        print("[ParserAgent] Nothing to parse")
        return {**state, "parsed_docs": [], "agents_used": state.get("agents_used", []) + ["parser_agent"]}

    s = get_settings()
    llm = ChatOllama(model=s.OLLAMA_FAST_MODEL, base_url=s.OLLAMA_BASE_URL, temperature=0)
    chain = EXTRACT_PROMPT | llm

    parsed: list[dict] = []
    comment_count = 0
    for doc in raw_docs:
        try:
            content_snippet = doc.get("content", "")[:800]
            response = chain.invoke({"content": content_snippet})
            meta = extract_json(response.content)

            timestamp = doc.get("timestamp", time.time())
            time_window = epoch_to_window(float(timestamp))
            source = doc.get("subreddit") or _extract_domain(doc.get("url", ""))

            comments = _parse_comment_tree(doc.get("comments", []), doc.get("url", ""), source)
            comment_count += _count_comments(comments)

            parsed.append({
                "id": _stable_post_id(doc),
                "title": doc.get("title", ""),
                "body": doc.get("content", ""),
                "url": doc.get("url", ""),
                "author": doc.get("author", "unknown"),
                "subreddit_or_source": source,
                "created_utc": float(timestamp),
                "time_window": time_window,
                "score": doc.get("score", 0),
                "source_type": doc.get("source_type", "web"),
                "topics": meta.get("topics", []),
                "entities": meta.get("entities", []),
                "sentiment_label": (meta.get("sentiment") or {}).get("label", "neutral"),
                "sentiment_score": (meta.get("sentiment") or {}).get("score", 0.5),
                "comments": comments,
            })
        except Exception as e:
            print(f"[ParserAgent] Error parsing {doc.get('url','?')}: {e}")

    print(f"[ParserAgent] Parsed {len(parsed)} documents, {comment_count} comments")
    return {
        **state,
        "parsed_docs": parsed,
        "agents_used": state.get("agents_used", []) + ["parser_agent"],
    }


def _parse_comment_tree(nodes: list[dict], post_url: str, source: str) -> list[dict]:
    """Recursively attach RAG metadata to a thresholded comment tree (no per-comment LLM call)."""
    parsed: list[dict] = []
    for c in nodes:
        body = c.get("body") or ""
        created = float(c.get("created_utc", time.time()))
        parsed.append({
            "id": c.get("id", ""),
            "body": body,
            "author": c.get("author") or "unknown",
            "url": post_url,
            "subreddit_or_source": source,
            "created_utc": created,
            "time_window": epoch_to_window(created),
            "parent_comment_id": c.get("parent_comment_id"),
            "depth": c.get("depth", 0),
            "sentiment_label": "neutral",
            "sentiment_score": 0.5,
            "replies": _parse_comment_tree(c.get("replies", []), post_url, source),
        })
    return parsed


def _count_comments(nodes: list[dict]) -> int:
    return sum(1 + _count_comments(n.get("replies", [])) for n in nodes)


def _stable_post_id(doc: dict) -> str:
    """Stable, idempotent post id: real Reddit id when available, else a hash of the URL."""
    reddit_id = doc.get("reddit_id")
    if reddit_id:
        return f"reddit_{reddit_id}"
    url = doc.get("url", "")
    return "web_" + hashlib.sha1(url.encode("utf-8")).hexdigest()[:16]


def _extract_domain(url: str) -> str:
    try:
        from urllib.parse import urlparse
        return urlparse(url).netloc.replace("www.", "")
    except Exception:
        return "web"
