from typing import Final
from langchain.schema import SystemMessage, HumanMessage
from app.llm_pipeline import format_messages_as_prompt
from app.llm_config import get_chat_model

# Define a set of valid intents for user input classification
VALID_INTENTS: Final[set[str]] = {"search", "summary", "qa", "recommend"}

def detect_user_intent(user_input: str) -> str:
    """
    Detect and classify the user's input into one of the predefined intents:
      - search: user wants to search for papers
      - summary: user wants a summary of a paper
      - qa: user asks a question about the content of a paper
      - recommend: user wants related papers recommended

    Args:
        user_input (str): The user's input message.

    Returns:
        str: A lowercase keyword representing the detected intent.
             Defaults to 'search' if the model output is invalid.
    """
    # Prepare the messages for the intent classification model
    messages = [
        SystemMessage(content="You are an intent classification assistant."),
        HumanMessage(content=(
            "Classify the following input into exactly one category:\n"
            "- search: user wants to search for papers\n"
            "- summary: user wants a summary of a paper\n"
            "- qa: user asks a question about the content of a paper\n"
            "- recommend: user wants related papers recommended\n"
            "Only return one keyword: search, summary, qa, or recommend.\n"
            f"Input: {user_input}"
        ))
    ]
    # Format the messages into a prompt for the model
    prompt = format_messages_as_prompt(messages)
    # Invoke the chat model to classify the intent
    result = get_chat_model().invoke(prompt)

    # Extract the intent from the model's response
    if hasattr(result, "content"):
        intent = result.content.strip().lower()
    elif isinstance(result, str):
        intent = result.strip().lower()
    else:
        intent = str(result).strip().lower()

    # Return the intent if valid, otherwise default to 'search'
    return intent if intent in VALID_INTENTS else "search"