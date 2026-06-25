"""Agent 3 — Reddit historical data via Pullpush.io (no auth needed)."""
import asyncio
import time
import httpx
from agents.state import AgentState
from utils.config import get_settings

PULLPUSH_BASE = "https://api.pullpush.io"
SUBREDDITS = ["MachineLearning", "LocalLLaMA", "artificial", "ChatGPT"]

# Comment selection thresholds (by upvote score)
COMMENT_TOP_N = 5     # top-level comments kept per post
REPLY_TOP_N = 3       # replies kept per comment, applied at every depth
COMMENT_MAX_DEPTH = 8  # safety guard against runaway recursion


def run_crawler_agent(state: AgentState) -> AgentState:
    intent = state.get("intent", {})
    topics = intent.get("topics", [])
    search_term = " ".join(topics[:2]) if topics else ""

    content = asyncio.run(_fetch_pullpush_data(search_term))
    valid = [c for c in content if c.get("content") and len(c["content"]) > 80]
    print(f"[CrawlerAgent] Collected {len(valid)} documents")

    return {
        **state,
        "crawled_content": valid,
        "agents_used": state.get("agents_used", []) + ["crawler_agent"],
    }


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


async def _pullpush_get(client: httpx.AsyncClient, path: str, params: dict, retries: int = 2) -> list[dict]:
    """GET a Pullpush endpoint with simple retry/backoff; returns the `data` list (or [])."""
    last_err = None
    for attempt in range(retries + 1):
        try:
            resp = await client.get(f"{PULLPUSH_BASE}{path}", params=params)
            resp.raise_for_status()
            return resp.json().get("data", [])
        except Exception as e:
            last_err = e
            await asyncio.sleep(0.8 * (attempt + 1))
    raise last_err if last_err else RuntimeError("pullpush request failed")


async def _pullpush_submissions(
    client: httpx.AsyncClient,
    subreddit: str,
    after: int,
    before: int,
    search_term: str,
    size: int = 15,
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
        posts = await _pullpush_get(client, "/reddit/search/submission/", params)
    except Exception as e:
        print(f"[CrawlerAgent] Pullpush submissions error ({subreddit}): {type(e).__name__}: {e}")
        return []

    # Fetch comments for all posts concurrently (bounded) instead of one-at-a-time.
    sem = asyncio.Semaphore(10)

    async def _comments_for(pid: str) -> list[dict]:
        async with sem:
            return await _pullpush_comments(client, pid)

    comment_lists = await asyncio.gather(
        *[_comments_for(p.get("id", "")) for p in posts]
    )

    results = []
    for post, raw_comments in zip(posts, comment_lists):
        post_id = post.get("id", "")
        comment_tree = _build_comment_tree(raw_comments, post_id)
        body = post.get("selftext", "") or ""
        content = f"Title: {post.get('title','')}\n\n{body}"

        results.append({
            "reddit_id": post_id,
            "url": f"https://reddit.com{post.get('permalink','')}",
            "title": post.get("title", ""),
            "content": content,
            "author": post.get("author", "unknown"),
            "timestamp": float(post.get("created_utc", time.time())),
            "source_type": "reddit",
            "subreddit": subreddit,
            "score": post.get("score", 0),
            "comments": comment_tree,
        })
    return results


async def _pullpush_comments(
    client: httpx.AsyncClient, post_id: str, size: int = 60
) -> list[dict]:
    """Fetch a flat list of raw comment dicts for a post (used to rebuild the tree)."""
    if not post_id:
        return []
    try:
        rows = await _pullpush_get(
            client,
            "/reddit/search/comment/",
            {"link_id": post_id, "size": size, "sort_type": "score", "sort": "desc"},
        )
        return [
            c for c in rows
            if c.get("body") not in ("[deleted]", "[removed]", None) and c.get("id")
        ]
    except Exception as e:
        print(f"[CrawlerAgent] Pullpush comments error ({post_id}): {type(e).__name__}: {e}")
        return []


def _build_comment_tree(raw_comments: list[dict], post_id: str) -> list[dict]:
    """Reconstruct a thresholded comment tree from a flat Pullpush comment list.

    Keeps the top COMMENT_TOP_N top-level comments by score, then recursively the
    top REPLY_TOP_N replies at each depth (capped at COMMENT_MAX_DEPTH).
    Threading uses Reddit's parent_id prefixes: ``t3_`` = the post, ``t1_`` = a comment.
    """
    children: dict[str, list[dict]] = {}
    for c in raw_comments:
        parent = c.get("parent_id") or ""
        children.setdefault(parent, []).append(c)

    def node(c: dict, depth: int) -> dict:
        cid = c.get("id", "")
        replies: list[dict] = []
        if depth < COMMENT_MAX_DEPTH:
            kids = sorted(
                children.get(f"t1_{cid}", []),
                key=lambda x: x.get("score", 0),
                reverse=True,
            )[:REPLY_TOP_N]
            replies = [node(k, depth + 1) for k in kids]
        return {
            "id": cid,
            "body": c.get("body", ""),
            "author": c.get("author", "unknown"),
            "score": c.get("score", 0),
            "created_utc": float(c.get("created_utc", time.time())),
            "parent_comment_id": None if depth == 0 else c.get("parent_id", "")[3:] or None,
            "depth": depth,
            "replies": replies,
        }

    top_level = sorted(
        children.get(f"t3_{post_id}", []),
        key=lambda x: x.get("score", 0),
        reverse=True,
    )[:COMMENT_TOP_N]
    return [node(c, 0) for c in top_level]
