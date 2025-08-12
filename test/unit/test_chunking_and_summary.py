import pytest
from langchain.docstore.document import Document
from app.vector_store import build_vector_db_from_texts, load_vector_db
from app.multi_summary import summarize_documents
from app.config import RETRIEVAL_TOP_K


def test_build_and_retrieve_with_chunking(tmp_path, monkeypatch):
    """Test that a long text is chunked, stored in FAISS, and retrieved correctly."""
    # Use a temporary FAISS directory
    monkeypatch.setenv("FAISS_TMP_DIR", str(tmp_path))

    # Build DB with a long-ish text (forces chunking)
    long_text = ("Transformer architecture improves NLP tasks. " * 300)
    build_vector_db_from_texts([long_text], save_path=str(tmp_path / "db"))

    # Load and query the DB
    db = load_vector_db(str(tmp_path / "db"))
    results = db.similarity_search("Transformer", k=RETRIEVAL_TOP_K)
    assert len(results) > 0, "No results retrieved from vector DB."


def test_summarize_documents_chunks():
    """Test summarization of multiple large documents."""
    docs = [
        Document(page_content="Paper A: proposes a new transformer variant. " * 50),
        Document(page_content="Paper B: sparse attention for efficiency. " * 50),
    ]
    summary = summarize_documents(docs)
    assert isinstance(summary, str), "Summary output should be a string."
    assert len(summary.strip()) > 0, "Summary is unexpectedly empty."
