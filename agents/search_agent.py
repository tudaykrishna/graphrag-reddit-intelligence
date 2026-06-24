"""Agent 2 — Brave Search API for discovering discussion URLs."""
import httpx
from urllib.parse import quote_plus
from agents.state import AgentState
from utils.config import get_settings

SKIP_DOMAINS = {"google.com", "facebook.com", "twitter.com", "instagram.com",
                "youtube.com", "amazon.com", "wikipedia.org", "linkedin.com"}

DISCUSSION_SITES = ["reddit.com", "news.ycombinator.com", "stackoverflow.com",
                    "dev.to", "medium.com", "hackernews"]


def run_search_agent(state: AgentState) -> AgentState:
    intent = state.get("intent", {})
    topics = intent.get("topics", [])
    entities = intent.get("entities", [])
    query = state["query"]
    s = get_settings()

    sub_queries = _build_sub_queries(query, topics, entities)
    all_urls: list[str] = []

    with httpx.Client(timeout=15) as client:
        for sub_q in sub_queries:
            urls = _brave_search(client, sub_q, s.BRAVE_API_KEY)
            all_urls.extend(urls)

    unique_urls = _deduplicate(all_urls)
    print(f"[SearchAgent] Found {len(unique_urls)} URLs for {len(sub_queries)} sub-queries")

    return {
        **state,
        "search_urls": unique_urls,
        "agents_used": state.get("agents_used", []) + ["search_agent"],
    }


def _build_sub_queries(original: str, topics: list[str], entities: list[str]) -> list[str]:
    queries = [f"{original} discussion forum"]
    for topic in topics[:2]:
        queries.append(f"{topic} reddit discussion community")
    if entities:
        queries.append(f"{' '.join(entities[:2])} community opinions reddit")
    return queries[:3]


def _brave_search(client: httpx.Client, query: str, api_key: str, count: int = 10) -> list[str]:
    if not api_key:
        print("[SearchAgent] No BRAVE_API_KEY — skipping web search")
        return []
    try:
        response = client.get(
            "https://api.search.brave.com/res/v1/web/search",
            params={"q": query, "count": count},
            headers={"Accept": "application/json", "X-Subscription-Token": api_key},
        )
        response.raise_for_status()
        data = response.json()
        results = data.get("web", {}).get("results", [])
        return [r["url"] for r in results if r.get("url")]
    except Exception as e:
        print(f"[SearchAgent] Brave search error: {e}")
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
