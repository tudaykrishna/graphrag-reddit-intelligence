from typing import Any, Optional
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from graph.neo4j_client import get_neo4j_client


class Neo4jGraphRetriever(BaseRetriever):
    time_window: Optional[str] = None
    k: int = 10

    model_config = {"arbitrary_types_allowed": True}

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        client = get_neo4j_client()
        keywords = [w.lower() for w in query.split() if len(w) > 3][:5]

        window_filter = ""
        params: dict[str, Any] = {"k": self.k}
        if self.time_window:
            window_filter = "AND p.time_window = $time_window"
            params["time_window"] = self.time_window

        docs = []
        for kw in keywords:
            params["kw"] = f"(?i).*{kw}.*"
            cypher = f"""
            MATCH (p:Post)
            WHERE p.title =~ $kw OR p.body =~ $kw
            {window_filter}
            RETURN p.id AS id, p.title AS title, p.body AS body,
                   p.url AS url, p.author AS author, p.subreddit_or_source AS source,
                   p.created_utc AS created_utc, p.time_window AS time_window,
                   p.sentiment_label AS sentiment
            LIMIT $k
            """
            rows = client.run(cypher, **params)
            for r in rows:
                text = f"{r.get('title','')}\n{r.get('body','')}"
                docs.append(Document(
                    page_content=text,
                    metadata={
                        "id": r.get("id", ""),
                        "source": r.get("source", ""),
                        "author": r.get("author", ""),
                        "url": r.get("url", ""),
                        "created_utc": r.get("created_utc", 0),
                        "time_window": r.get("time_window", ""),
                        "sentiment": r.get("sentiment", ""),
                        "retriever": "graph",
                    },
                ))

        seen = set()
        unique = []
        for d in docs:
            if d.metadata["id"] not in seen:
                seen.add(d.metadata["id"])
                unique.append(d)
        return unique[: self.k]


def get_topic_evolution(topic: str) -> list[dict]:
    client = get_neo4j_client()
    cypher = """
    MATCH (t:Topic {name: $topic})<-[r:DISCUSSES]-(p:Post)
    RETURN p.time_window AS window, count(p) AS mentions
    ORDER BY window
    """
    return client.run(cypher, topic=topic)


def get_influential_users(time_window: Optional[str] = None) -> list[dict]:
    client = get_neo4j_client()
    filter_clause = "WHERE p.time_window = $tw" if time_window else ""
    params = {"tw": time_window} if time_window else {}
    cypher = f"""
    MATCH (u:User)<-[:AUTHORED_BY]-(p:Post)
    {filter_clause}
    WITH u.username AS user, count(p) AS posts
    MATCH (u2:User {{username: user}})<-[:AUTHORED_BY]-(c:Comment)
    RETURN user, posts, count(c) AS comments, posts + count(c) AS activity
    ORDER BY activity DESC
    LIMIT 10
    """
    return client.run(cypher, **params)


def get_topic_counts_by_window() -> list[dict]:
    client = get_neo4j_client()
    cypher = """
    MATCH (t:Topic)<-[r:DISCUSSES]-(p:Post)
    RETURN t.name AS topic, p.time_window AS window, count(p) AS count
    ORDER BY count DESC
    LIMIT 50
    """
    return client.run(cypher)


def check_topic_coverage(topics: list[str]) -> int:
    if not topics:
        return 0
    client = get_neo4j_client()
    total = 0
    for topic in topics[:3]:
        rows = client.run(
            "MATCH (t:Topic {name: $name}) RETURN count(t) AS c", name=topic.lower()
        )
        total += rows[0]["c"] if rows else 0
    return total
