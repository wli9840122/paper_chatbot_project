# tests/test_intent_classifier.py
import pytest
from app.intent_classifier import detect_user_intent

# --- Simple stub chat model that returns a fixed .content ---
class _StubChatModel:
    def __init__(self, reply: str):
        self._reply = reply
    def invoke(self, _payload, **_kwargs):
        # mimic an AIMessage-like object with `.content`
        return type("Msg", (), {"content": self._reply})

@pytest.fixture(autouse=True)
def setup_env(monkeypatch, tmp_path):
    # Keep sessions & models isolated for tests
    monkeypatch.setenv("SESSION_FILE", str(tmp_path / ".chat_sessions.json"))
    monkeypatch.setenv("USE_LOCAL_MODEL", "true")
    monkeypatch.setenv("ALLOW_ONLINE_FALLBACK", "false")

def test_detect_user_intent_valid_outputs(monkeypatch):
    # Force LLM to return each valid intent; ensure it passes the whitelist.
    import app.intent_classifier as ic
    for out in ["search", "summary", "qa", "recommend"]:
        monkeypatch.setattr(ic, "get_chat_model", lambda: _StubChatModel(out), raising=True)
        assert detect_user_intent("dummy") == out

def test_detect_user_intent_sanitize(monkeypatch):
    # When model returns noisy text, we should sanitize to 'search'
    import app.intent_classifier as ic
    monkeypatch.setattr(ic, "get_chat_model", lambda: _StubChatModel("SEARCH and more text"), raising=True)
    assert detect_user_intent("Find papers") == "search"

def test_detect_user_intent_default_fallback(monkeypatch):
    # When model returns an invalid token, default to 'search'
    import app.intent_classifier as ic
    monkeypatch.setattr(ic, "get_chat_model", lambda: _StubChatModel("other"), raising=True)
    assert detect_user_intent("Whatever") == "search"
