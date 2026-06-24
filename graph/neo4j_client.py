from neo4j import GraphDatabase
from functools import lru_cache
from utils.config import get_settings


class Neo4jClient:
    def __init__(self):
        s = get_settings()
        self._driver = GraphDatabase.driver(s.NEO4J_URI, auth=(s.NEO4J_USER, s.NEO4J_PASSWORD))

    def close(self):
        self._driver.close()

    def run(self, query: str, **params) -> list[dict]:
        with self._driver.session() as session:
            result = session.run(query, **params)
            return [dict(record) for record in result]

    def run_write(self, query: str, **params) -> None:
        with self._driver.session() as session:
            session.execute_write(lambda tx: tx.run(query, **params))

    def batch_write(self, query: str, rows: list[dict]) -> None:
        with self._driver.session() as session:
            session.execute_write(lambda tx: tx.run(query, rows=rows))


@lru_cache(maxsize=1)
def get_neo4j_client() -> Neo4jClient:
    return Neo4jClient()
