"""Agent 4 — Parse raw crawled content, extract entities/topics/sentiment via LLM."""
import uuid
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
    for doc in raw_docs:
        try:
            content_snippet = doc.get("content", "")[:800]
            response = chain.invoke({"content": content_snippet})
            meta = extract_json(response.content)

            timestamp = doc.get("timestamp", time.time())
            time_window = epoch_to_window(float(timestamp))

            parsed.append({
                "id": str(uuid.uuid4()),
                "title": doc.get("title", ""),
                "body": doc.get("content", ""),
                "url": doc.get("url", ""),
                "author": doc.get("author", "unknown"),
                "subreddit_or_source": doc.get("subreddit") or _extract_domain(doc.get("url", "")),
                "created_utc": float(timestamp),
                "time_window": time_window,
                "score": doc.get("score", 0),
                "source_type": doc.get("source_type", "web"),
                "topics": meta.get("topics", []),
                "entities": meta.get("entities", []),
                "sentiment_label": meta.get("sentiment", {}).get("label", "neutral"),
                "sentiment_score": meta.get("sentiment", {}).get("score", 0.5),
            })
        except Exception as e:
            print(f"[ParserAgent] Error parsing {doc.get('url','?')}: {e}")

    print(f"[ParserAgent] Parsed {len(parsed)} documents")
    return {
        **state,
        "parsed_docs": parsed,
        "agents_used": state.get("agents_used", []) + ["parser_agent"],
    }


def _extract_domain(url: str) -> str:
    try:
        from urllib.parse import urlparse
        return urlparse(url).netloc.replace("www.", "")
    except Exception:
        return "web"
