# tests/test_vector_store.py
import pytest
from app.vector_store import build_vector_db_from_texts, load_vector_db

def test_vector_db(tmp_path):
    """Test that we can build and load a FAISS vector DB."""
    sample_texts = [
        "This is a test document.",
        "Another one with content."
    ]
    db_path = tmp_path / "test_db"

    # Build DB
    build_vector_db_from_texts(sample_texts, save_path=str(db_path))

    # Load DB
    db = load_vector_db(str(db_path))
    assert db is not None
