# tests/test_session_manager.py
import json
import uuid
import pytest
from app import session_manager as sm

@pytest.fixture
def temp_session_file(tmp_path, monkeypatch):
    """Ensure session data is stored in a temp location for testing."""
    session_file = tmp_path / ".chat_sessions.json"
    monkeypatch.setattr(sm, "SESSION_FILE", str(session_file))
    return session_file

def test_create_and_retrieve_chat_id(temp_session_file):
    # Initially, file doesn't exist
    assert not temp_session_file.exists()

    # Create chat_id
    chat_id = sm.get_or_create_chat_id("user1")
    assert isinstance(chat_id, str)
    assert uuid.UUID(chat_id)  # valid UUID format

    # File should now exist with correct mapping
    assert temp_session_file.exists()
    data = json.loads(temp_session_file.read_text())
    assert data["user1"] == chat_id

    # Calling again returns same chat_id
    same_chat_id = sm.get_or_create_chat_id("user1")
    assert same_chat_id == chat_id

def test_multiple_users(temp_session_file):
    id1 = sm.get_or_create_chat_id("userA")
    id2 = sm.get_or_create_chat_id("userB")
    assert id1 != id2

    data = json.loads(temp_session_file.read_text())
    assert set(data.keys()) == {"userA", "userB"}

def test_reset_chat_id(temp_session_file):
    old_id = sm.get_or_create_chat_id("userX")
    new_id = sm.reset_chat_id("userX")
    assert new_id != old_id
    assert uuid.UUID(new_id)

    data = json.loads(temp_session_file.read_text())
    assert data["userX"] == new_id

def test_load_sessions_corrupted_file(temp_session_file):
    # Simulate corrupted JSON
    temp_session_file.write_text("{invalid json}")
    sessions = sm._load_sessions()
    assert sessions == {}

