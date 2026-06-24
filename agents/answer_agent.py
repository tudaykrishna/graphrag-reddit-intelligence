"""Agent 6 — Hybrid retrieval (graph + vector, RRF via EnsembleRetriever) + LLM answer."""
from langchain.retrievers import EnsembleRetriever
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from agents.state import AgentState
from graph.graph_retriever import Neo4jGraphRetriever, get_topic_evolution
from vector.vector_store import get_retriever
from utils.config import get_settings

ANSWER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a discussion intelligence assistant. Answer questions based on the provided discussion context.
Always cite your sources using [Source: url, date] format.
Be specific and reference actual content from the discussions."""),
    ("human", """Question: {question}

Retrieved Discussion Context:
{context}

Provide a comprehensive answer with source citations."""),
])

COMPARE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "You are a temporal analysis assistant comparing discussion trends across time periods."),
    ("human", """Question: {question}

Period 1 Context ({window1}):
{context1}

Period 2 Context ({window2}):
{context2}

Compare these two time periods. What changed? What stayed the same? Highlight key trends and shifts."""),
])


def run_answer_agent(state: AgentState) -> AgentState:
    query = state["query"]
    intent = state.get("intent", {})
    s = get_settings()

    time_window = intent.get("time_range", {}).get("label") if intent.get("time_range") else None
    is_comparison = intent.get("is_temporal_comparison", False)
    compare_windows = intent.get("compare_windows", [])

    llm = ChatOllama(model=s.OLLAMA_SMART_MODEL, base_url=s.OLLAMA_BASE_URL, temperature=0.2)

    if is_comparison and len(compare_windows) >= 2:
        answer, graph_res, vec_res = _temporal_comparison(query, compare_windows, llm)
    else:
        answer, graph_res, vec_res = _hybrid_query(query, time_window, llm)

    sources = _extract_sources(graph_res + vec_res)

    print(f"[AnswerAgent] Graph results: {len(graph_res)}, Vector results: {len(vec_res)}")
    print(f"[AnswerAgent] Generated answer ({len(answer)} chars)")

    return {
        **state,
        "answer": answer,
        "graph_results": [{"content": d.page_content[:300], "metadata": d.metadata} for d in graph_res],
        "vector_results": [{"content": d.page_content[:300], "metadata": d.metadata} for d in vec_res],
        "sources": sources,
        "agents_used": state.get("agents_used", []) + ["answer_agent"],
    }


def _hybrid_query(query: str, time_window: str | None, llm: ChatOllama) -> tuple[str, list, list]:
    graph_retriever = Neo4jGraphRetriever(time_window=time_window, k=8)
    vector_retriever = get_retriever(time_window=time_window, top_k=8)

    ensemble = EnsembleRetriever(
        retrievers=[graph_retriever, vector_retriever],
        weights=[0.5, 0.5],
    )

    graph_docs = graph_retriever.get_relevant_documents(query)
    vector_docs = vector_retriever.get_relevant_documents(query)
    fused_docs = ensemble.get_relevant_documents(query)

    context = _format_context(fused_docs[:8])
    chain = ANSWER_PROMPT | llm | StrOutputParser()
    answer = chain.invoke({"question": query, "context": context})

    return answer, graph_docs, vector_docs


def _temporal_comparison(query: str, windows: list[str], llm: ChatOllama) -> tuple[str, list, list]:
    w1, w2 = windows[0], windows[1]

    gr1 = Neo4jGraphRetriever(time_window=w1, k=5)
    vr1 = get_retriever(time_window=w1, top_k=5)
    ens1 = EnsembleRetriever(retrievers=[gr1, vr1], weights=[0.5, 0.5])
    docs1 = ens1.get_relevant_documents(query)

    gr2 = Neo4jGraphRetriever(time_window=w2, k=5)
    vr2 = get_retriever(time_window=w2, top_k=5)
    ens2 = EnsembleRetriever(retrievers=[gr2, vr2], weights=[0.5, 0.5])
    docs2 = ens2.get_relevant_documents(query)

    chain = COMPARE_PROMPT | llm | StrOutputParser()
    answer = chain.invoke({
        "question": query,
        "window1": w1,
        "context1": _format_context(docs1[:5]),
        "window2": w2,
        "context2": _format_context(docs2[:5]),
    })

    return answer, docs1, docs2


def _format_context(docs: list[Document]) -> str:
    parts = []
    for i, doc in enumerate(docs, 1):
        meta = doc.metadata
        source = meta.get("source", "")
        url = meta.get("url", "")
        created = meta.get("created_utc", 0)
        from datetime import datetime, timezone
        date_str = datetime.fromtimestamp(float(created), tz=timezone.utc).strftime("%Y-%m-%d") if created else "unknown"
        parts.append(f"[{i}] Source: {source} | {url} | {date_str}\n{doc.page_content[:400]}")
    return "\n\n".join(parts)


def _extract_sources(docs: list[Document]) -> list[dict]:
    seen = set()
    sources = []
    for doc in docs:
        url = doc.metadata.get("url", "")
        if url and url not in seen:
            seen.add(url)
            sources.append({
                "url": url,
                "source": doc.metadata.get("source", ""),
                "author": doc.metadata.get("author", ""),
                "time_window": doc.metadata.get("time_window", ""),
            })
    return sources
