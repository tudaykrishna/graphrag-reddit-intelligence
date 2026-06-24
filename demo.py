"""CLI demo — runs 4 queries through the full multi-agent pipeline."""
from agents.orchestrator import run_pipeline

DEMO_QUERIES = [
    ("SEMANTIC", "What technical challenges do people face when fine-tuning LLMs locally?"),
    ("GRAPH TRAVERSAL", "Who are the most active voices discussing AI safety and what communities are they in?"),
    ("HYBRID", "How has community sentiment around open-source AI models evolved?"),
    ("TEMPORAL COMPARISON", "Compare AI discussions in Q4 2024 vs Q1 2025 — what topics emerged or faded?"),
]

DIVIDER = "=" * 70


def print_result(query_type: str, query: str, state: dict) -> None:
    print(f"\n{DIVIDER}")
    print(f"  QUERY TYPE : {query_type}")
    print(f"  QUERY      : {query}")
    print(DIVIDER)

    print(f"\n  Agents used : {' → '.join(state.get('agents_used', []))}")
    print(f"  RAG had data: {state.get('has_rag_data', False)}")
    print(f"  New docs ingested: {state.get('ingested', False)}")

    graph_results = state.get("graph_results", [])
    print(f"\n  --- Graph Results ({len(graph_results)}) ---")
    for i, r in enumerate(graph_results[:3], 1):
        content = r.get("content", "")[:150].replace("\n", " ")
        meta = r.get("metadata", {})
        print(f"  [{i}] {meta.get('source','?')} | {meta.get('time_window','?')} | {content}...")

    vector_results = state.get("vector_results", [])
    print(f"\n  --- Vector Results ({len(vector_results)}) ---")
    for i, r in enumerate(vector_results[:3], 1):
        content = r.get("content", "")[:150].replace("\n", " ")
        meta = r.get("metadata", {})
        print(f"  [{i}] {meta.get('source','?')} | {meta.get('time_window','?')} | {content}...")

    print(f"\n  --- Fused Answer ---")
    answer = state.get("answer", "No answer generated")
    for line in answer.split("\n"):
        print(f"  {line}")

    sources = state.get("sources", [])
    if sources:
        print(f"\n  --- Sources ({len(sources)}) ---")
        for s in sources[:5]:
            print(f"  • {s.get('url','')} [{s.get('time_window','')}]")


def main():
    print(f"\n{DIVIDER}")
    print("  GraphRAG Discussion Intelligence — Multi-Agent Demo")
    print(DIVIDER)
    print("  This demo runs 4 queries through the full agent pipeline.")
    print("  Each query may trigger: search → crawl → parse → ingest → answer")
    print(f"{DIVIDER}\n")

    for query_type, query in DEMO_QUERIES:
        print(f"\n[Running {query_type}] {query}")
        try:
            state = run_pipeline(query)
            print_result(query_type, query, state)
        except Exception as e:
            print(f"  ERROR: {e}")

    print(f"\n{DIVIDER}")
    print("  Demo complete.")
    print(DIVIDER)


if __name__ == "__main__":
    main()
