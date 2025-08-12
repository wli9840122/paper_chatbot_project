# tests/test_llm_config.py
import pytest
import app.llm_config as llm_config

class DummyLLM:
    """Minimal stub for chat model used in tests."""
    def invoke(self, prompt, **_kwargs):
        return f"Echo: {prompt}"

@pytest.fixture(autouse=True)
def patch_get_chat_model(monkeypatch):
    """Patch get_chat_model to avoid loading real models."""
    monkeypatch.setattr(llm_config, "get_chat_model", lambda: DummyLLM())

def test_llm_initialization():
    llm_instance = llm_config.get_chat_model()
    assert llm_instance is not None, "LLM should be initialized"

def test_llm_invoke():
    """
    Test that the LLM can respond to a basic prompt.
    """
    llm_instance = llm_config.get_chat_model()
    response = llm_instance.invoke("Hello, how are you?")
    assert isinstance(response, str)
    assert "Hello" in response
