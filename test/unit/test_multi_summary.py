# tests/test_multi_summary.py
import pytest
from langchain.docstore.document import Document
import app.multi_summary as multi_summary

class DummyLLM:
    """Stub chat model for summarization tests."""
    def invoke(self, prompt, **_kwargs):
        return "This is a fake summary."

@pytest.fixture(autouse=True)
def patch_get_chat_model(monkeypatch):
    """Patch get_chat_model in multi_summary to avoid heavy model loading."""
    monkeypatch.setattr(multi_summary, "get_chat_model", lambda: DummyLLM())

def test_summarize_documents_output():
    docs = [
        Document(page_content="This paper introduces a novel attention mechanism for transformers."),
        Document(page_content="We propose a method to reduce model size while maintaining accuracy.")
    ]
    summary = multi_summary.summarize_documents(docs)
    assert isinstance(summary, str)
    assert len(summary.strip()) > 0
    assert "summary" in summary.lower() or "fake" in summary.lower()
