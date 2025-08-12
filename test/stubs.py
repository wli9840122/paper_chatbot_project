from langchain_core.messages import AIMessage

class StubChatModel:
    """Minimal .invoke() to stand in for a chat model."""
    def __init__(self, responses=None):
        self.responses = responses or {}
    def invoke(self, payload, **kwargs):
        # payload may be {"messages":[...]} or str
        text = ""
        if isinstance(payload, dict):
            msgs = payload.get("messages") or []
            text = msgs[-1][1] if msgs else ""
        elif isinstance(payload, str):
            text = payload
        out = self.responses.get(text, "ok")
        return AIMessage(content=out)