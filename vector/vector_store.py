from functools import lru_cache
from typing import Optional
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever
from utils.config import get_settings

COLLECTION_NAME = "reddit_discussions"


@lru_cache(maxsize=1)
def _get_embeddings() -> HuggingFaceEmbeddings:
    s = get_settings()
    return HuggingFaceEmbeddings(model_name=s.EMBEDDING_MODEL)


@lru_cache(maxsize=1)
def get_vector_store() -> Chroma:
    s = get_settings()
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=_get_embeddings(),
        persist_directory=s.CHROMA_PERSIST_DIR,
    )


def add_documents(docs: list[Document], ids: Optional[list[str]] = None) -> None:
    store = get_vector_store()
    # Passing stable ids makes Chroma upsert (re-ingesting the same content won't duplicate).
    store.add_documents(docs, ids=ids) if ids else store.add_documents(docs)


def get_retriever(time_window: Optional[str] = None, top_k: int = 10) -> VectorStoreRetriever:
    store = get_vector_store()
    search_kwargs: dict = {"k": top_k}
    if time_window:
        search_kwargs["filter"] = {"time_window": time_window}
    return store.as_retriever(search_kwargs=search_kwargs)


def similarity_search_with_score(query: str, top_k: int = 10) -> list[tuple[Document, float]]:
    store = get_vector_store()
    return store.similarity_search_with_score(query, k=top_k)


def count_documents() -> int:
    store = get_vector_store()
    return store._collection.count()
