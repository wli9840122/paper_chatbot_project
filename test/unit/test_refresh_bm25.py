# tests/test_bm25_cache.py
import pytest
from app.vector_store import build_vector_db_from_texts, load_vector_db as _load_db

def test_refresh_bm25_cache(tmp_path, monkeypatch):
    # Build a FAISS DB in a temp dir (avoid touching real data)
    vector_dir = tmp_path / "db"
    vector_dir.mkdir(parents=True, exist_ok=True)

    # FAISS on Windows sometimes needs this
    monkeypatch.setenv("FAISS_TMP_DIR", str(tmp_path))

    # Write DB explicitly to our temp path
    build_vector_db_from_texts(["Test doc about transformers."], save_path=str(vector_dir))

    # Make retriever_factory load from our temp DB regardless of its internal default
    import app.retriever_factory as rf
    monkeypatch.setattr(rf, "load_vector_db", lambda: _load_db(str(vector_dir)), raising=True)

    # Now exercise the BM25 cache + retriever
    from app.retriever_factory import refresh_bm25_cache, get_retriever
    refresh_bm25_cache()

    retriever = get_retriever("transformers")
    docs = retriever.get_relevant_documents("transformers")
    assert isinstance(docs, list) and len(docs) > 0

