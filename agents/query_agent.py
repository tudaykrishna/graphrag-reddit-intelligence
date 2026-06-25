"""Agent 1 — Query Understanding + RAG Coverage Check."""
import json
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from agents.state import AgentState
from graph.graph_retriever import check_topic_coverage
from vector.vector_store import similarity_search_with_score, count_documents
from utils.config import get_settings
from utils.json_parser import extract_json
from utils.temporal import parse_comparison_windows

PARSE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "You are a query analysis assistant. Extract structured information from user queries."),
    ("human", """Analyze this query and return ONLY a JSON object:

Query: {query}

Return this exact JSON structure:
{{
  "topics": ["topic1", "topic2"],
  "entities": ["entity1", "entity2"],
  "time_range": null,
  "is_temporal_comparison": false,
  "compare_windows": []
}}

- topics: main discussion themes (2-4 short phrases)
- entities: specific named things (models, companies, people, tools)
- time_range: null or {{"label": "Q1_2025"}} if a specific time period is mentioned
- is_temporal_comparison: true if comparing two time periods
- compare_windows: list of time window labels if comparing (e.g. ["Q4_2024", "Q1_2025"])

Return ONLY the JSON."""),
])


def run_query_agent(state: AgentState) -> AgentState:
    query = state["query"]
    s = get_settings()
    llm = ChatOllama(model=s.OLLAMA_SMART_MODEL, base_url=s.OLLAMA_BASE_URL, temperature=0)

    chain = PARSE_PROMPT | llm
    response = chain.invoke({"query": query})
    intent = extract_json(response.content)

    if not intent.get("topics"):
        intent["topics"] = [query[:50]]

    compare_windows = parse_comparison_windows(query)
    if compare_windows:
        intent["compare_windows"] = compare_windows
        intent["is_temporal_comparison"] = True

    has_rag_data = _check_coverage(query, intent)

    print(f"[QueryAgent] Intent: {json.dumps(intent, indent=2)}")
    print(f"[QueryAgent] RAG has data: {has_rag_data}")

    return {
        **state,
        "intent": intent,
        "has_rag_data": has_rag_data,
        "agents_used": state.get("agents_used", []) + ["query_agent"],
    }


def _check_coverage(query: str, intent: dict) -> bool:
    total_docs = count_documents()
    if total_docs < 10:
        return False

    results = similarity_search_with_score(query, top_k=8)
    high_score = [r for r in results if r[1] < 0.8]
    if len(high_score) >= 3:
        return True

    topics = intent.get("topics", [])
    graph_count = check_topic_coverage(topics)
    return graph_count >= 3
