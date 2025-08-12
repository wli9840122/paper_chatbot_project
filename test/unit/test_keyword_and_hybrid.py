# tests/test_keyword_and_retriever.py
import pytest
from app.keyword_extractor import extract_search_keyword
from app.retriever_factory import get_retriever
from app.vector_store import build_vector_db_from_texts

class DummyLLM:
    """Stub LLM that just returns a canned keyword."""
    def invoke(self, prompt, **_kwargs):
        # Simulate AIMessage-like object
        return type("Msg", (), {"content": "graph neural networks"})

def test_keyword_extraction_with_llm(monkeypatch):
    # Force LLM keyword extraction path
    monkeypatch.setenv("USE_LLM_KEYWORD", "true")
    kw = extract_search_keyword("Papers about graph neural networks", llm=DummyLLM())
    assert isinstance(kw, str)
    assert "graph" in kw.lower()

def test_hybrid_search(tmp_path, monkeypatch):
    # Create a small FAISS DB in a temp dir
    texts = [
        "Linformer reduces self-attention complexity.",
        "BERT is used for NLP tasks.",
        "Graph neural networks process graph data."
    ]
    build_vector_db_from_texts(texts, save_path=str(tmp_path))

    # Point the retriever to the temp DB
    import app.vector_store as vs
    monkeypatch.setattr(vs, "VECTOR_DIR", str(tmp_path))

    retriever = get_retriever("graph neural network")
    docs = retriever.get_relevant_documents("graph neural network")
    assert isinstance(docs, list)
    assert len(docs) > 0
