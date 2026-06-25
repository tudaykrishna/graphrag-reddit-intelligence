from typing import TypedDict, Optional


class AgentState(TypedDict, total=False):
    query: str
    intent: dict                   # {topics, entities, time_range, compare_windows}
    has_rag_data: bool
    search_urls: list[str]
    crawled_content: list[dict]    # [{url, title, content, author, timestamp, source_type, comments: [nested tree]}]
    parsed_docs: list[dict]        # [{id, title, body, url, author, created_utc, time_window, topics, entities, sentiment_label, comments: [nested tree]}]
    ingested: bool
    graph_results: list[dict]      # raw graph retrieval results
    vector_results: list[dict]     # raw vector retrieval results
    answer: str
    sources: list[dict]
    agents_used: list[str]
    error: Optional[str]
