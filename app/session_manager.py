import os
import uuid
import json

SESSION_FILE = os.getenv("SESSION_FILE", ".chat_sessions.json")  # Path to the session file, defaulting to ".chat_sessions.json"

def _load_sessions():
    """
    Load session data from the session file.

    Returns:
        dict: A dictionary containing session data. If the file does not exist or an error occurs, returns an empty dictionary.
    """
    if not os.path.exists(SESSION_FILE):
        # Return an empty dictionary if the session file does not exist
        return {}
    try:
        # Attempt to read and parse the session file as JSON
        with open(SESSION_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        # Return an empty dictionary if an error occurs during file reading or parsing
        return {}

def _save_sessions(sessions):
    """
    Save session data to the session file.

    Args:
        sessions (dict): A dictionary containing session data to be saved.
    """
    with open(SESSION_FILE, "w", encoding="utf-8") as f:
        # Write the session data to the file in JSON format with indentation
        json.dump(sessions, f, indent=2)

def get_or_create_chat_id(user_identifier: str = "default") -> str:
    """
    Retrieve an existing chat ID for a user or create a new one if it doesn't exist.

    Args:
        user_identifier (str): A unique identifier for the user, such as a username, IP address, or browser session ID. Defaults to "default".

    Returns:
        str: The chat ID associated with the user.
    """
    sessions = _load_sessions()  # Load existing session data
    if user_identifier not in sessions:
        # Generate a new chat ID if the user does not have an existing session
        sessions[user_identifier] = str(uuid.uuid4())
        _save_sessions(sessions)  # Save the updated session data
    return sessions[user_identifier]  # Return the chat ID for the user

def reset_chat_id(user_identifier: str):
    """
    Reset the chat ID for a user, forcing the start of a new conversation.

    Args:
        user_identifier (str): A unique identifier for the user, such as a username, IP address, or browser session ID.

    Returns:
        str: The new chat ID associated with the user.
    """
    sessions = _load_sessions()  # Load existing session data
    # Generate a new chat ID for the user
    sessions[user_identifier] = str(uuid.uuid4())
    _save_sessions(sessions)  # Save the updated session data
    return sessions[user_identifier]  # Return the new chat ID for the user