from fastapi import APIRouter, Query
from graph.graph_retriever import (
    get_topic_counts_by_window,
    get_influential_users,
    get_topic_evolution,
)
from vector.vector_store import count_documents

router = APIRouter()


@router.get("/graph/topics")
def topics_endpoint(time_window: str = Query(None, description="e.g. Q1_2025")):
    data = get_topic_counts_by_window()
    if time_window:
        data = [d for d in data if d.get("window") == time_window]
    return {"topics": data, "time_window": time_window}


@router.get("/graph/users")
def users_endpoint(time_window: str = Query(None)):
    users = get_influential_users(time_window)
    return {"users": users, "time_window": time_window}


@router.get("/graph/topic/{topic}/evolution")
def topic_evolution_endpoint(topic: str):
    evolution = get_topic_evolution(topic)
    return {"topic": topic, "evolution": evolution}


@router.get("/stats")
def stats_endpoint():
    vector_count = count_documents()
    return {
        "vector_documents": vector_count,
        "status": "ok",
    }
