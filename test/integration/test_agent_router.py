# tests/integration/test_llm_pipeline.py
import pytest
from langchain_core.messages import AIMessage

class StubChatModel:
    def invoke(self, payload, **kwargs):
        # behave like a chat model and return an AIMessage
        return AIMessage(content="stub answer")

@pytest.fixture(autouse=True)
def patch_chat_model(monkeypatch):
    # Ensure no real model loads
    import app.llm_config as llm_config
    monkeypatch.setattr(llm_config, "get_chat_model", lambda: StubChatModel(), raising=True)
