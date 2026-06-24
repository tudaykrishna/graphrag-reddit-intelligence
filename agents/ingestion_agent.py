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

    all_chunks = []
    for doc in docs:
        chunks = chunk_post(doc)
        all_chunks.extend(chunks)

    if all_chunks:
        add_documents(all_chunks)
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
        source = doc.get("subreddit_or_source", "unknown")
        author = doc.get("author", "unknown")
        time_window = doc.get("time_window", "")
        created_utc = doc.get("created_utc", 0.0)
        post_id = doc.get("id", "")

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
            title=doc.get("title", "")[:500],
            body=doc.get("body", "")[:2000],
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
