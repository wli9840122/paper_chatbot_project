# tests/integration/test_llm_pipeline.py
import pytest
from langchain_core.messages import AIMessage

class StubChatModel:
    def invoke(self, payload, **kwargs):
        text = payload if isinstance(payload, str) else str(payload)
        return AIMessage(content=f"[stub answer] {text[:60]}")

@pytest.fixture(autouse=True)
def patch_chat_model(monkeypatch):
    import app.llm_config as llm_config
    monkeypatch.setattr(llm_config, "get_chat_model", lambda: StubChatModel(), raising=True)

@pytest.mark.network
def test_ask_question_with_rag_basic():
    from app.llm_pipeline import ask_question_with_rag
    context = [
        "This paper proposes a novel Transformer architecture for sequence modeling.",
        "It achieves state-of-the-art results on multiple NLP benchmarks.",
    ]
    out = ask_question_with_rag(
        question="What is the main contribution of the paper?",
        context_docs=context,
        chat_id="t-basic",
    )
    assert isinstance(out, str) and out.strip()
    assert "[stub answer]" in out

@pytest.mark.network
def test_ask_question_with_rag_persists_history():
    import app.llm_pipeline as lp
    ctx = ["A small context paragraph about transformers."]
    chat_id = "t-history"

    _ = lp.ask_question_with_rag("Q1?", ctx, chat_id=chat_id)
    _ = lp.ask_question_with_rag("Q2?", ctx, chat_id=chat_id)

    assert chat_id in lp.chat_memories
    mem = lp.chat_memories[chat_id]
    msgs = getattr(mem, "chat_memory", mem).messages
    assert len(msgs) >= 4
