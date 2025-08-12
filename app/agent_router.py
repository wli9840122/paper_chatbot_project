"""
LangGraph-based agent router.
- Tools: search, summarize (from last search), QA, recommend.
- Uses app.llm_config.get_chat_model() to lazily init the chat LLM.
- Persists multi-turn conversation per user via LangGraph MemorySaver (thread_id).
- Uses LangGraph prebuilt ReAct agent.
"""
from datetime import date
from typing import List, Dict, Any
from langchain_core.tools import Tool
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver

from app.keyword_extractor import extract_search_keyword
from app.arxiv_search import search_arxiv
from app.multi_summary import summarize_documents
from app.vector_store import load_vector_db
from app.recommender import recommend_similar_papers
from app.session_manager import get_or_create_chat_id
from app.llm_pipeline import ask_question_with_rag
from app.llm_config import get_chat_model
import arxiv

# Single process-wide memory store; LangGraph isolates by thread_id we pass per user/session
_MEMORY = MemorySaver()

# Cache for agents per chat_id
_agent_cache: Dict[str, Any] = {}

# Cache last search results per chat session (chat_id -> List[Document])
_LAST_SEARCH_RESULTS: Dict[str, List] = {}

# -----------------------------------------------------------------------------------
# Tool Functions
# -----------------------------------------------------------------------------------

def tool_search_arxiv(query: str, *, chat_id: str | None = None) -> str:
    """
    Search ArXiv for AI/ML papers and return short summaries/links.
    Caches results for the current chat session if chat_id is provided.

    Args:
        query (str): The search query.
        chat_id (str | None): The chat session ID for caching results.

    Returns:
        str: A formatted string of search results or a message if no papers are found.
    """
    llm = get_chat_model()
    keyword = extract_search_keyword(query, llm=llm)

    # Detect "latest" intent -> sort by date; otherwise relevance
    q_lower = query.lower()
    recency_words = ["latest", "newest", "recent", "this year", "past year", "last year"]
    want_latest = any(w in q_lower for w in recency_words)

    since = None
    sort_by = arxiv.SortCriterion.Relevance
    if want_latest:
        sort_by = arxiv.SortCriterion.SubmittedDate
        if "this year" in q_lower:
            since = str(date.today().year)
        elif "past year" in q_lower or "last year" in q_lower:
            since = str(date.today().year - 1)

    docs = search_arxiv(
        keyword,
        max_results=5,
        sort_by=sort_by,
        since=since
    )
    if not docs:
        return "No papers found."

    # Cache for this chat if chat_id provided by wrapper
    if chat_id is not None:
        _LAST_SEARCH_RESULTS[chat_id] = docs

    # Pretty print results with indices
    lines = []
    for i, d in enumerate(docs, 1):
        title = d.metadata.get("title", "Untitled")
        authors = d.metadata.get("authors", "Unknown authors")
        pub_date = d.metadata.get("published", "Unknown date")
        url = d.metadata.get("url", "")
        lines.append(f"{i}. 📄 {title}\n   👤 {authors} | 📅 {pub_date}\n   🔗 {url}")

    lines.append("\nTip: To summarize, call SummarizePaper with indices like '1,3' (or leave empty to summarize all).")
    return "\n".join(lines)


def _summarize_wrapper(chat_id: str):
    """
    Create a wrapper function for summarizing papers from the last search.

    Args:
        chat_id (str): The chat session ID.

    Returns:
        Callable: A function that summarizes papers based on indices.
    """
    def _summarize(indices_str: str) -> str:
        docs = _LAST_SEARCH_RESULTS.get(chat_id) or []
        if not docs:
            return "No papers in context. Run SearchArXiv first."

        # Parse indices like "1" or "1,3"
        indices_str = (indices_str or "").strip()
        if indices_str:
            try:
                idxs = [int(x) for x in indices_str.replace(" ", "").split(",") if x]
                picked = [docs[i - 1] for i in idxs if 1 <= i <= len(docs)]
                if not picked:
                    return "No valid indices. Use a comma-separated list like '1,3'."
            except Exception:
                return "Invalid indices. Use a comma-separated list like '1,3'."
        else:
            picked = docs  # Summarize all

        return summarize_documents(picked)
    return _summarize


def tool_qa(query: str, chat_id: str) -> str:
    """
    Answer questions using the vector database (RAG).

    Args:
        query (str): The user query.
        chat_id (str): The chat session ID.

    Returns:
        str: The answer to the query or an error message if the vector DB is unavailable.
    """
    try:
        db = load_vector_db()
    except Exception as e:
        return f"Vector DB not available: {e}"
    ctx_docs = db.similarity_search(query, k=3)
    context = [d.page_content for d in ctx_docs]
    return ask_question_with_rag(query, context, chat_id)


def tool_recommend(query: str) -> str:
    """
    Recommend related papers based on a topic or seed query.

    Args:
        query (str): The topic or seed query.

    Returns:
        str: A formatted string of recommended papers.
    """
    return recommend_similar_papers(query, k_papers=5, fetch_k=60)


# Wrap tool_qa and search to bind chat_id at runtime
def _qa_wrapper(chat_id: str):
    """
    Create a wrapper function for the QA tool, binding the chat_id.

    Args:
        chat_id (str): The chat session ID.

    Returns:
        Callable: A function that performs QA with the given chat_id.
    """
    return lambda q: tool_qa(q, chat_id)

def _search_wrapper(chat_id: str):
    """
    Create a wrapper function for the search tool, binding the chat_id.

    Args:
        chat_id (str): The chat session ID.

    Returns:
        Callable: A function that performs search with the given chat_id.
    """
    return lambda q: tool_search_arxiv(q, chat_id=chat_id)

# -----------------------------------------------------------------------------------
# LangGraph Agent (prebuilt ReAct) + Memory
# -----------------------------------------------------------------------------------

def _build_tools(chat_id: str) -> List[Tool]:
    """
    Create the tool list for the agent.

    Args:
        chat_id (str): The chat session ID.

    Returns:
        List[Tool]: A list of tools available to the agent.
    """
    return [
        Tool(
            name="SearchArXiv",
            func=_search_wrapper(chat_id),
            description="Search ArXiv for AI/ML papers. Caches results for this chat and shows a numbered list."
        ),
        Tool(
            name="SummarizePaper",
            func=_summarize_wrapper(chat_id),
            description="Summarize papers from the LAST SearchArXiv in this chat. "
                        "Pass indices like '1,3' to pick specific items; leave empty to summarize all."
        ),
        Tool(
            name="PaperQA",
            func=_qa_wrapper(chat_id),
            description="Answer a question based on paper content already indexed in the vector DB."
        ),
        Tool(
            name="RecommendPapers",
            func=tool_recommend,
            description="Recommend at most 5 related papers based on a topic or seed query."
        ),
    ]


def _get_agent(chat_id: str):
    """
    Retrieve or create a LangGraph agent for the given chat_id.

    Args:
        chat_id (str): The chat session ID.

    Returns:
        Any: The LangGraph agent instance.
    """
    if chat_id not in _agent_cache:
        model = get_chat_model()                 # Chat model (LangChain Runnable)
        tools = _build_tools(chat_id)
        _agent_cache[chat_id] = create_react_agent(
            model=model,
            tools=tools,
            checkpointer=_MEMORY
        )
    return _agent_cache[chat_id]

# -----------------------------------------------------------------------------------
# Public Entry Point
# -----------------------------------------------------------------------------------

def run_agent(user_input: str, user_identifier: str = "default") -> str:
    """
    Route user input via LangGraph agent with per-user memory.

    Args:
        user_input (str): The user's input message.
        user_identifier (str): The user identifier, mapped to a persistent chat_id.

    Returns:
        str: The latest assistant message text.
    """
    chat_id = get_or_create_chat_id(user_identifier)
    agent = _get_agent(chat_id)
    config = {"configurable": {"thread_id": chat_id}}
    result = agent.invoke({"messages": [("user", user_input)]}, config=config)

    try:
        messages = result.get("messages") or []
        if messages and hasattr(messages[-1], "content"):
            return messages[-1].content
        if isinstance(result, dict) and "output" in result:
            return str(result["output"])
        return str(result)
    except Exception:
        return str(result)