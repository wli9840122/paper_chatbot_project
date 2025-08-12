import pytest
from app.arxiv_search import search_arxiv

def test_arxiv_search_returns_documents():
    """Ensure ArXiv search returns documents with required metadata."""
    query = "transformer"
    docs = search_arxiv(query, max_results=3)

    assert docs, "No documents returned from ArXiv search."
    for doc in docs:
        assert "title" in doc.metadata, "Missing 'title' in document metadata."
        assert "url" in doc.metadata, "Missing 'url' in document metadata."
        assert isinstance(doc.page_content, str), "page_content is not a string."
