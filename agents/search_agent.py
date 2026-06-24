"""Agent 2 — DuckDuckGo search for discovering discussion URLs (no API key needed)."""
from duckduckgo_search import DDGS
from agents.state import AgentState

SKIP_DOMAINS = {"google.com", "facebook.com", "twitter.com", "instagram.com",
                "youtube.com", "amazon.com", "wikipedia.org", "linkedin.com"}


def run_search_agent(state: AgentState) -> AgentState:
    intent = state.get("intent", {})
    topics = intent.get("topics", [])
    entities = intent.get("entities", [])
    query = state["query"]

    sub_queries = _build_sub_queries(query, topics, entities)
    all_urls: list[str] = []

    with DDGS() as ddgs:
        for sub_q in sub_queries:
            urls = _ddg_search(ddgs, sub_q)
            all_urls.extend(urls)

    unique_urls = _deduplicate(all_urls)
    print(f"[SearchAgent] Found {len(unique_urls)} URLs for {len(sub_queries)} sub-queries")

    return {
        **state,
        "search_urls": unique_urls,
        "agents_used": state.get("agents_used", []) + ["search_agent"],
    }


def _build_sub_queries(original: str, topics: list[str], entities: list[str]) -> list[str]:
    queries = [f"{original} reddit discussion"]
    for topic in topics[:2]:
        queries.append(f"{topic} reddit community opinions")
    if entities:
        queries.append(f"{' '.join(entities[:2])} reddit discussion forum")
    return queries[:3]


def _ddg_search(ddgs: DDGS, query: str, max_results: int = 8) -> list[str]:
    try:
        results = ddgs.text(query, max_results=max_results)
        return [r["href"] for r in results if r.get("href")]
    except Exception as e:
        print(f"[SearchAgent] DuckDuckGo error: {e}")
        return []


def _deduplicate(urls: list[str]) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for url in urls:
        domain = _extract_domain(url)
        if domain in SKIP_DOMAINS:
            continue
        if url not in seen:
            seen.add(url)
            unique.append(url)
    return unique[:15]


def _extract_domain(url: str) -> str:
    try:
        from urllib.parse import urlparse
        return urlparse(url).netloc.replace("www.", "")
    except Exception:
        return ""
