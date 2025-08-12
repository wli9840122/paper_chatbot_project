import os
from typing import Optional
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
from app.vector_store import load_vector_db
from app.keyword_extractor import extract_search_keyword
from app.llm_config import get_chat_model

# Config
USE_HYBRID = os.getenv("USE_HYBRID", "true").lower() == "true"  # Flag to enable hybrid retrieval
BM25_K = int(os.getenv("BM25_K", 8))  # Number of top results to retrieve using BM25
VECTOR_K = int(os.getenv("VECTOR_K", 8))  # Number of top results to retrieve using vector search
HYBRID_WEIGHTS = [float(w) for w in os.getenv("HYBRID_WEIGHTS", "0.4,0.6").split(",")]  # Weights for hybrid retriever

_bm25_cache: Optional[BM25Retriever] = None  # Cache for BM25 retriever


def build_bm25_retriever(docs: list[Document]) -> BM25Retriever:
    """
    Build a BM25 retriever from a list of documents.

    Args:
        docs (list[Document]): A list of documents to index.

    Returns:
        BM25Retriever: The BM25 retriever instance.
    """
    bm25 = BM25Retriever.from_documents(docs)  # Create BM25 retriever from documents
    bm25.k = BM25_K  # Set the number of top results to retrieve
    return bm25


def get_retriever(query: str) -> EnsembleRetriever | FAISS:
    """
    Get a retriever based on the configuration.

    Args:
        query (str): The search query.

    Returns:
        EnsembleRetriever | FAISS: A hybrid retriever (BM25 + Vector) or a vector-only retriever.
    """
    vector_db = load_vector_db()  # Load the vector database
    vector_retriever = vector_db.as_retriever(search_kwargs={"k": VECTOR_K})  # Create a vector retriever

    if not USE_HYBRID:
        # Return vector-only retriever if hybrid retrieval is disabled
        return vector_retriever

    global _bm25_cache
    if _bm25_cache is None:
        # Build BM25 retriever if not already cached
        docs = list(vector_db.docstore._dict.values())  # Retrieve all documents from the vector database
        _bm25_cache = build_bm25_retriever(docs)

    # Extract search keyword using the LLM
    llm = get_chat_model()  # Get the chat model instance
    keyword = extract_search_keyword(query, llm=llm)  # Extract keywords from the query
    print(f"[HybridSearch] Using keyword: {keyword}")

    # Return a hybrid retriever combining BM25 and vector retrievers
    return EnsembleRetriever(
        retrievers=[
            _bm25_cache,  # BM25 retriever with keyword optimization
            vector_retriever  # Vector retriever
        ],
        weights=HYBRID_WEIGHTS  # Weights for combining the retrievers
    )


def refresh_bm25_cache():
    """
    Refresh the BM25 retriever cache by rebuilding it from the latest vector database documents.
    Call this function after ingesting new documents.
    """
    global _bm25_cache
    try:
        vector_db = load_vector_db()  # Load the vector database
        docs = list(vector_db.docstore._dict.values())  # Retrieve all documents from the vector database
        _bm25_cache = build_bm25_retriever(docs)  # Rebuild the BM25 retriever
        print("[HybridSearch] BM25 cache refreshed.")
    except Exception as e:
        # Log an error message if the cache refresh fails
        print(f"[HybridSearch] Failed to refresh BM25 cache: {e}")