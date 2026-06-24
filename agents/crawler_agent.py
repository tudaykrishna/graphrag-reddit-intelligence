"""Agent 3 — Crawl URLs via Crawl4AI; Reddit historical data via Pullpush.io (no auth needed)."""
import asyncio
import time
import re
from urllib.parse import urlparse
import httpx
from crawl4ai import AsyncWebCrawler
from agents.state import AgentState
from utils.config import get_settings

PULLPUSH_BASE = "https://api.pullpush.io"
SUBREDDITS = ["MachineLearning", "LocalLLaMA", "artificial", "ChatGPT"]


def run_crawler_agent(state: AgentState) -> AgentState:
    urls = state.get("search_urls", [])
    intent = state.get("intent", {})

    content = asyncio.run(_crawl_all(urls, intent))
    valid = [c for c in content if c.get("content") and len(c["content"]) > 80]
    print(f"[CrawlerAgent] Collected {len(valid)} documents")

    return {
        **state,
        "crawled_content": valid,
        "agents_used": state.get("agents_used", []) + ["crawler_agent"],
    }


async def _crawl_all(urls: list[str], intent: dict) -> list[dict]:
    reddit_urls = [u for u in urls if "reddit.com" in u]
    other_urls = [u for u in urls if "reddit.com" not in u]

    topics = intent.get("topics", [])
    search_term = " ".join(topics[:2]) if topics else ""

    tasks = []
    if reddit_urls:
        tasks.append(_crawl_reddit_urls(reddit_urls))
    if other_urls:
        tasks.append(_crawl_web_batch(other_urls))
    # Always pull some historical Reddit data via Pullpush
    tasks.append(_fetch_pullpush_data(search_term))

    results = await asyncio.gather(*tasks, return_exceptions=True)
    combined: list[dict] = []
    for r in results:
        if isinstance(r, list):
            combined.extend(r)
    return combined


async def _crawl_reddit_urls(urls: list[str]) -> list[dict]:
    """Scrape Reddit post pages with Crawl4AI (Playwright renders JS)."""
    results = []
    async with AsyncWebCrawler(verbose=False) as crawler:
        tasks = [_crawl_single(crawler, url) for url in urls[:8]]
        crawled = await asyncio.gather(*tasks, return_exceptions=True)
        for item in crawled:
            if isinstance(item, dict):
                item["source_type"] = "reddit"
                results.append(item)
    return results


async def _fetch_pullpush_data(search_term: str) -> list[dict]:
    """Fetch posts + comments from Pullpush.io across all tracked subreddits."""
    from utils.temporal import TIME_WINDOWS
    results = []

    async with httpx.AsyncClient(timeout=20) as client:
        for sub in SUBREDDITS[:2]:
            for window_label, (after, before) in TIME_WINDOWS.items():
                posts = await _pullpush_submissions(client, sub, after, before, search_term)
                results.extend(posts)
                # Rate-limit friendliness
                await asyncio.sleep(0.5)

    print(f"[CrawlerAgent] Pullpush fetched {len(results)} Reddit documents")
    return results


async def _pullpush_submissions(
    client: httpx.AsyncClient,
    subreddit: str,
    after: int,
    before: int,
    search_term: str,
    size: int = 25,
) -> list[dict]:
    params: dict = {
        "subreddit": subreddit,
        "after": after,
        "before": before,
        "size": size,
        "sort_type": "score",
        "sort": "desc",
    }
    if search_term:
        params["q"] = search_term

    try:
        resp = await client.get(f"{PULLPUSH_BASE}/reddit/search/submission/", params=params)
        resp.raise_for_status()
        data = resp.json()
        posts = data.get("data", [])
    except Exception as e:
        print(f"[CrawlerAgent] Pullpush submissions error ({subreddit}): {e}")
        return []

    results = []
    for post in posts:
        post_id = post.get("id", "")
        comments_text = await _pullpush_comments(client, post_id)
        body = post.get("selftext", "") or ""
        content = f"Title: {post.get('title','')}\n\n{body}\n\nComments:\n" + "\n---\n".join(comments_text[:15])

        results.append({
            "url": f"https://reddit.com{post.get('permalink','')}",
            "title": post.get("title", ""),
            "content": content,
            "author": post.get("author", "unknown"),
            "timestamp": float(post.get("created_utc", time.time())),
            "source_type": "reddit",
            "subreddit": subreddit,
            "score": post.get("score", 0),
        })
    return results


async def _pullpush_comments(
    client: httpx.AsyncClient, post_id: str, size: int = 20
) -> list[str]:
    if not post_id:
        return []
    try:
        resp = await client.get(
            f"{PULLPUSH_BASE}/reddit/search/comment/",
            params={"link_id": post_id, "size": size, "sort_type": "score", "sort": "desc"},
        )
        resp.raise_for_status()
        data = resp.json()
        return [
            f"u/{c.get('author','?')}: {c.get('body','')}"
            for c in data.get("data", [])
            if c.get("body") not in ("[deleted]", "[removed]", None)
        ]
    except Exception:
        return []


async def _crawl_web_batch(urls: list[str]) -> list[dict]:
    results = []
    async with AsyncWebCrawler(verbose=False) as crawler:
        tasks = [_crawl_single(crawler, url) for url in urls[:8]]
        crawled = await asyncio.gather(*tasks, return_exceptions=True)
        for item in crawled:
            if isinstance(item, dict):
                results.append(item)
    return results


async def _crawl_single(crawler: AsyncWebCrawler, url: str) -> dict | None:
    try:
        result = await crawler.arun(url=url)
        if not result.success:
            return None
        markdown = result.markdown or ""
        title = ""
        if result.metadata:
            title = result.metadata.get("title", "")
        domain = urlparse(url).netloc.replace("www.", "")
        return {
            "url": url,
            "title": title,
            "content": markdown[:5000],
            "author": "web",
            "timestamp": time.time(),
            "source_type": "web",
            "subreddit": domain,
            "score": 0,
        }
    except Exception as e:
        print(f"[CrawlerAgent] Crawl error {url}: {e}")
        return None
