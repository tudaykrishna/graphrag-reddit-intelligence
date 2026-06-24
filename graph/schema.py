from graph.neo4j_client import get_neo4j_client

CONSTRAINTS = [
    "CREATE CONSTRAINT post_id IF NOT EXISTS FOR (p:Post) REQUIRE p.id IS UNIQUE",
    "CREATE CONSTRAINT comment_id IF NOT EXISTS FOR (c:Comment) REQUIRE c.id IS UNIQUE",
    "CREATE CONSTRAINT user_name IF NOT EXISTS FOR (u:User) REQUIRE u.username IS UNIQUE",
    "CREATE CONSTRAINT source_name IF NOT EXISTS FOR (s:Source) REQUIRE s.name IS UNIQUE",
    "CREATE CONSTRAINT topic_name IF NOT EXISTS FOR (t:Topic) REQUIRE t.name IS UNIQUE",
]

INDEXES = [
    "CREATE INDEX post_time IF NOT EXISTS FOR (p:Post) ON (p.created_utc)",
    "CREATE INDEX post_window IF NOT EXISTS FOR (p:Post) ON (p.time_window)",
    "CREATE INDEX comment_time IF NOT EXISTS FOR (c:Comment) ON (c.created_utc)",
    "CREATE INDEX comment_window IF NOT EXISTS FOR (c:Comment) ON (c.time_window)",
]


def init_schema():
    client = get_neo4j_client()
    for stmt in CONSTRAINTS + INDEXES:
        try:
            client.run(stmt)
        except Exception as e:
            print(f"[schema] {e}")
    print("[schema] Neo4j schema initialized.")
