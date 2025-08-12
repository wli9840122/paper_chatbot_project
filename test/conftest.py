# tests/conftest.py
import os
import socket
import pytest

# --- Hard offline: never download models ---
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("USE_LOCAL_MODEL", "true")
os.environ.setdefault("ALLOW_ONLINE_FALLBACK", "false")
os.environ.setdefault("USE_LLM_KEYWORD", "false")  # avoid LLM/keybert

# --- Block network for all tests unless explicitly allowed ---
@pytest.fixture(autouse=True)
def _no_network(request, monkeypatch):
    if request.node.get_closest_marker("network"):
        return  # allow opt-in
    def guard(*a, **k): raise RuntimeError("Network disabled")
    monkeypatch.setattr(socket, "create_connection", guard)
    class _DenySocket(socket.socket):
        def __new__(cls, *a, **k): raise RuntimeError("Network disabled")
    monkeypatch.setattr(socket, "socket", _DenySocket)

# --- Stub chat model everywhere BEFORE anything calls get_chat_model() ---
from langchain_core.messages import AIMessage
class _StubChatModel:
    def invoke(self, payload, **kwargs):
        text = payload if isinstance(payload, str) else str(payload)
        return AIMessage(content=f"[stub answer] {text[:60]}")

@pytest.fixture(autouse=True)
def _stub_chat_model(monkeypatch):
    # Make any import of get_chat_model return the stub
    import app.llm_config as llm_config
    monkeypatch.setattr(llm_config, "get_chat_model", lambda: _StubChatModel(), raising=True)

# --- Replace HuggingFaceEmbeddings with tiny fake & stop all model loads ---
@pytest.fixture(autouse=True)
def _fake_embeddings(monkeypatch):
    try:
        from langchain_community.embeddings import FakeEmbeddings
    except Exception:
        from langchain_community.embeddings.fake import FakeEmbeddings  # older LC

    import app.vector_store as vs

    class _TinyFake(FakeEmbeddings):
        def __init__(self): super().__init__(size=16)

    # 1) if module already created a global embedding_model, replace it
    if hasattr(vs, "embedding_model"):
        monkeypatch.setattr(vs, "embedding_model", _TinyFake(), raising=False)

    # 2) Stop any future creation of real HF embeddings in this module
    def _no_download_embedding(*args, **kwargs): return _TinyFake()
    monkeypatch.setattr(vs, "HuggingFaceEmbeddings", _no_download_embedding, raising=False)

# --- Make FAISS disk I/O a no-op (huge speed win on Windows/CI) ---
@pytest.fixture(autouse=True)
def _fast_faiss(monkeypatch):
    import app.vector_store as vs
    _memory_store = {}  # path -> vectorstore instance

    def _save_local(self, path):
        _memory_store[str(path)] = self  # store in RAM only

    def _load_local(path, *a, **k):
        key = str(path)
        if key not in _memory_store:
            raise ValueError(f"Vector DB directory not found: {path}")
        return _memory_store[key]

    monkeypatch.setattr(vs.FAISS, "save_local", _save_local, raising=True)
    monkeypatch.setattr(vs.FAISS, "load_local", staticmethod(_load_local), raising=True)

# --- Stub arXiv search everywhere (avoid heavy XML/feed parsing) ---
@pytest.fixture(autouse=True)
def _stub_arxiv_search(monkeypatch):
    try:
        import app.arxiv_search as arx
    except Exception:
        return  # not imported in this test session

    from langchain.docstore.document import Document
    fake_docs = [
        Document(
            page_content="This paper introduces a tiny transformer for tests.",
            metadata={
                "title": "Tiny Transformer for Tests",
                "authors": "Alice; Bob",
                "published": "2024-01-02",
                "url": "https://example.org/ttft",
                "arxiv_id": "2401.00001",
            },
        )
    ]
    monkeypatch.setattr(arx, "search_arxiv", lambda *a, **k: fake_docs, raising=True)
