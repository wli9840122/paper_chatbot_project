import os
from typing import Optional

# Check if LLM-based keyword extraction is enabled via environment variable
USE_LLM_KEYWORD = os.getenv("USE_LLM_KEYWORD", "true").lower() == "true"

try:
    # Attempt to import KeyBERT for local keyword extraction
    from keybert import KeyBERT
except ImportError:
    # Handle the case where KeyBERT is not installed
    KeyBERT = None


def llm_extract_keyword(llm, user_input: str) -> Optional[str]:
    """
    Extract a keyword using a Language Learning Model (LLM).

    Args:
        llm: The LLM instance to use for keyword extraction.
        user_input (str): The input text from which to extract the keyword.

    Returns:
        Optional[str]: The extracted keyword or None if extraction fails.
    """
    prompt = (
        f"Extract the most important keyword or short phrase from this query:\n"
        f"'{user_input}'\n"
        "Only return the keyword(s), no explanations."
    )
    try:
        # Invoke the LLM with the generated prompt
        response = llm.invoke(prompt)
        # Extract and return the content from the LLM response
        if hasattr(response, "content"):
            return response.content.strip()
        elif isinstance(response, str):
            return response.strip()
    except Exception as e:
        # Log any errors that occur during LLM invocation
        print(f"[KeywordExtractor] LLM keyword extraction failed: {e}")
    return None


def local_extract_keyword(user_input: str) -> str:
    """
    Perform local keyword extraction as a fallback method.

    Args:
        user_input (str): The input text from which to extract the keyword.

    Returns:
        str: The extracted keyword or the original input if extraction fails.
    """
    # 1. Attempt keyword extraction using KeyBERT
    if KeyBERT is not None:
        try:
            kw_model = KeyBERT()
            # Extract the top keyword from the input text
            keywords = kw_model.extract_keywords(user_input, top_n=1)
            if keywords:
                return keywords[0][0]
        except Exception as e:
            # Log any errors that occur during KeyBERT extraction
            print(f"[KeywordExtractor] KeyBERT failed: {e}")

    # 2. Fallback to returning the original input text
    return user_input


def extract_search_keyword(user_input: str, llm=None) -> str:
    """
    Unified interface for keyword extraction, using LLM if enabled and available.

    Args:
        user_input (str): The input text from which to extract the keyword.
        llm: The LLM instance to use for keyword extraction (optional).

    Returns:
        str: The extracted keyword or the original input if extraction fails.
    """
    # Use LLM-based extraction if enabled and an LLM instance is provided
    if USE_LLM_KEYWORD and llm is not None:
        keyword = llm_extract_keyword(llm, user_input)
        if keyword:
            return keyword
    # Fallback to local keyword extraction
    return local_extract_keyword(user_input)