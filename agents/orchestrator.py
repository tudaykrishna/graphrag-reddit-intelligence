"""LangGraph orchestrator wiring all 6 agents into a conditional state machine."""
from langgraph.graph import StateGraph, END
from agents.state import AgentState
from agents.query_agent import run_query_agent
from agents.crawler_agent import run_crawler_agent
from agents.parser_agent import run_parser_agent
from agents.ingestion_agent import run_ingestion_agent
from agents.answer_agent import run_answer_agent
from graph.schema import init_schema


def _route_after_query(state: AgentState) -> str:
    if state.get("has_rag_data", False):
        print("[Orchestrator] RAG has data -> skipping to answer")
        return "answer"
    print("[Orchestrator] RAG lacks data -> triggering Reddit ingest pipeline")
    return "crawl"


def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("query_understanding", run_query_agent)
    graph.add_node("crawl", run_crawler_agent)
    graph.add_node("parse", run_parser_agent)
    graph.add_node("ingest", run_ingestion_agent)
    graph.add_node("answer", run_answer_agent)

    graph.set_entry_point("query_understanding")

    graph.add_conditional_edges(
        "query_understanding",
        _route_after_query,
        {"answer": "answer", "crawl": "crawl"},
    )

    graph.add_edge("crawl", "parse")
    graph.add_edge("parse", "ingest")
    graph.add_edge("ingest", "answer")
    graph.add_edge("answer", END)

    return graph.compile()


_app = None


def get_app():
    global _app
    if _app is None:
        try:
            init_schema()
        except Exception as e:
            print(f"[Orchestrator] Schema init warning: {e}")
        _app = build_graph()
    return _app


def run_pipeline(query: str) -> AgentState:
    app = get_app()
    initial_state: AgentState = {
        "query": query,
        "agents_used": [],
        "has_rag_data": False,
        "search_urls": [],
        "crawled_content": [],
        "parsed_docs": [],
        "ingested": False,
        "graph_results": [],
        "vector_results": [],
        "answer": "",
        "sources": [],
        "intent": {},
    }
    result = app.invoke(initial_state)
    return result
