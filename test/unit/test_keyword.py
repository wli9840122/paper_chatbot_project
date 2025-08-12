# tests/test_keyword.py
import os
import pytest
from app.keyword_extractor import extract_search_keyword

class _StubLLM:
    """Tiny stub that mimics an LLM returning a keyword via .invoke()."""
    def __init__(self, reply: str):
        self._reply = reply
    def invoke(self, _prompt, **_kwargs):
        # Return an object with a .content attribute (like AIMessage)
        return type("Msg", (), {"content": self._reply})

def test_extract_keyword_local(monkeypatch):
    """
    When USE_LLM_KEYWORD=false or no llm is provided,
    fall back to local extraction (KeyBERT if available, else original text).
    """
    monkeypatch.setenv("USE_LLM_KEYWORD", "false")
    kw = extract_search_keyword("Recent papers on graph neural networks in chemistry", llm=None)
    assert isinstance(kw, str)
    assert len(kw.strip()) > 0

def test_extract_keyword_llm(monkeypatch):
    """
    When USE_LLM_KEYWORD=true and an LLM is provided, use the LLM path.
    """
    monkeypatch.setenv("USE_LLM_KEYWORD", "true")
    stub = _StubLLM("graph neural networks")
    kw = extract_search_keyword("Recent papers on graph neural networks in chemistry", llm=stub)
    assert isinstance(kw, str)
    assert "graph" in kw.lower()

