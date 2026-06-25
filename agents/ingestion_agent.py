"""Agent 5 — Chunk parsed docs, embed into ChromaDB, build Neo4j graph nodes."""
from graph.neo4j_client import get_neo4j_client
from vector.chunker import chunk_post, chunk_comment
from vector.vector_store import add_documents
from agents.state import AgentState


def run_ingestion_agent(state: AgentState) -> AgentState:
    docs = state.get("parsed_docs", [])
    if not docs:
        print("[IngestionAgent] Nothing to ingest")
        return {**state, "ingested": False, "agents_used": state.get("agents_used", []) + ["ingestion_agent"]}

    # Build stable chunk ids and de-duplicate within the batch (the crawler can return
    # the same post/comment more than once; Chroma requires unique ids per upsert call).
    all_chunks = []
    chunk_ids = []
    seen_ids: set[str] = set()
    for doc in docs:
        for chunk in chunk_post(doc):
            cid = f"{chunk.metadata.get('id','')}:{chunk.metadata.get('chunk_index', 0)}"
            if cid in seen_ids:
                continue
            seen_ids.add(cid)
            all_chunks.append(chunk)
            chunk_ids.append(cid)
        for comment in _flatten_comments(doc.get("comments", [])):
            chunk = chunk_comment(comment)
            cid = f"c:{chunk.metadata.get('id','')}"
            if cid in seen_ids:
                continue
            seen_ids.add(cid)
            all_chunks.append(chunk)
            chunk_ids.append(cid)

    if all_chunks:
        add_documents(all_chunks, ids=chunk_ids)
        print(f"[IngestionAgent] Added {len(all_chunks)} chunks to ChromaDB")

    _write_to_neo4j(docs)
    print(f"[IngestionAgent] Wrote {len(docs)} docs to Neo4j")

    return {
        **state,
        "ingested": True,
        "agents_used": state.get("agents_used", []) + ["ingestion_agent"],
    }


def _write_to_neo4j(docs: list[dict]) -> None:
    client = get_neo4j_client()

    for doc in docs:
        source = doc.get("subreddit_or_source") or "unknown"
        author = doc.get("author") or "unknown"
        time_window = doc.get("time_window") or ""
        created_utc = doc.get("created_utc") or 0.0
        post_id = doc.get("id") or ""

        client.run_write("""
            MERGE (s:Source {name: $source})
            MERGE (u:User {username: $author})
            MERGE (p:Post {id: $id})
            SET p.title = $title,
                p.body = $body,
                p.url = $url,
                p.score = $score,
                p.created_utc = $created_utc,
                p.time_window = $time_window,
                p.sentiment_label = $sentiment,
                p.subreddit_or_source = $source,
                p.author = $author
            MERGE (p)-[:FROM_SOURCE {created_at: $created_utc}]->(s)
            MERGE (p)-[:AUTHORED_BY {created_at: $created_utc}]->(u)
            MERGE (u)-[:ACTIVE_IN {time_window: $time_window}]->(s)
        """,
            id=post_id,
            title=(doc.get("title") or "")[:500],
            body=(doc.get("body") or "")[:2000],
            url=doc.get("url", ""),
            score=doc.get("score", 0),
            created_utc=created_utc,
            time_window=time_window,
            sentiment=doc.get("sentiment_label", "neutral"),
            source=source,
            author=author,
        )

        for topic_name in doc.get("topics", []):
            if topic_name:
                client.run_write("""
                    MERGE (t:Topic {name: $name})
                    WITH t
                    MATCH (p:Post {id: $post_id})
                    MERGE (p)-[:DISCUSSES {created_at: $created_utc, time_window: $tw}]->(t)
                """, name=topic_name.lower(), post_id=post_id, created_utc=created_utc, tw=time_window)

        for ent in doc.get("entities", []):
            name = ent.get("name", "")
            etype = ent.get("type", "CONCEPT")
            if name:
                client.run_write("""
                    MERGE (e:Entity {name: $name, type: $type})
                    WITH e
                    MATCH (p:Post {id: $post_id})
                    MERGE (p)-[:MENTIONS {created_at: $created_utc, time_window: $tw}]->(e)
                """, name=name, type=etype, post_id=post_id, created_utc=created_utc, tw=time_window)

        _write_comment_tree(client, doc.get("comments", []), post_id, None, source)


def _flatten_comments(nodes: list[dict]) -> list[dict]:
    """Depth-first flatten of a nested comment tree into a single list."""
    flat: list[dict] = []
    for n in nodes:
        flat.append(n)
        flat.extend(_flatten_comments(n.get("replies", [])))
    return flat


def _write_comment_tree(
    client, comments: list[dict], post_id: str, parent_comment_id, source: str
) -> None:
    """Recursively MERGE Comment nodes and their AUTHORED_BY / COMMENTS_ON / REPLY_TO edges."""
    for c in comments:
        comment_id = c.get("id") or ""
        if not comment_id:
            continue
        author = c.get("author") or "unknown"
        time_window = c.get("time_window") or ""
        created_utc = c.get("created_utc") or 0.0

        client.run_write("""
            MERGE (u:User {username: $author})
            MERGE (c:Comment {id: $id})
            SET c.body = $body,
                c.score = $score,
                c.created_utc = $created_utc,
                c.time_window = $time_window,
                c.sentiment_label = $sentiment,
                c.depth = $depth,
                c.url = $url,
                c.author = $author,
                c.subreddit_or_source = $source
            MERGE (c)-[:AUTHORED_BY {created_at: $created_utc}]->(u)
            WITH c, u
            MATCH (s:Source {name: $source})
            MERGE (u)-[:ACTIVE_IN {time_window: $time_window}]->(s)
        """,
            id=comment_id,
            body=(c.get("body") or "")[:2000],
            score=c.get("score", 0),
            created_utc=created_utc,
            time_window=time_window,
            sentiment=c.get("sentiment_label", "neutral"),
            depth=c.get("depth", 0),
            url=c.get("url", ""),
            author=author,
            source=source,
        )

        if parent_comment_id:
            client.run_write("""
                MATCH (c:Comment {id: $id})
                MATCH (parent:Comment {id: $parent_id})
                MERGE (c)-[:REPLY_TO {created_at: $created_utc}]->(parent)
            """, id=comment_id, parent_id=parent_comment_id, created_utc=created_utc)
        else:
            client.run_write("""
                MATCH (c:Comment {id: $id})
                MATCH (p:Post {id: $post_id})
                MERGE (c)-[:COMMENTS_ON {created_at: $created_utc}]->(p)
            """, id=comment_id, post_id=post_id, created_utc=created_utc)

        _write_comment_tree(client, c.get("replies", []), post_id, comment_id, source)
