import os
import re
from typing import List

# Define a set of stopwords to filter out common, non-informative words
STOPWORDS = set("""
a an the and or for to of in on with about from as by at into over under between within without
paper papers recent latest new approach method methods model models study studies
""".split())

def _clean_terms(terms: List[str]) -> List[str]:
    """
    Clean and filter a list of terms by removing duplicates, stopwords, and invalid entries.

    Args:
        terms (List[str]): A list of terms to clean.

    Returns:
        List[str]: A cleaned list of terms, limited to a maximum of 3 unique terms.
    """
    out, seen = [], set()
    for t in terms:
        # Normalize the term: lowercase, remove non-alphanumeric characters, and trim whitespace
        t = re.sub(r"[^\w\s\-.]", " ", t.lower()).strip()
        t = re.sub(r"\s+", " ", t)
        # Skip terms that are empty, in the stopwords list, or too short
        if not t or t in STOPWORDS or len(t) < 2:
            continue
        # Add the term to the output if it hasn't been seen before
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out[:3]  # Return at most 3 terms

def extract_keyword(user_input: str) -> str:
    """
    Extract up to 3 research keywords from the user input using multiple strategies:
      1) LLM-based extraction (preferred)
      2) KeyBERT-based extraction (fallback)
      3) Regex-based extraction (final fallback)

    Args:
        user_input (str): The input text from which to extract keywords.

    Returns:
        str: A space-separated string of up to 3 extracted keywords.
    """
    # 1) LLM keyword extraction
    try:
        from transformers import pipeline
        # Load the LLM model specified in the environment variable or use the default model
        model_name = os.getenv("KEYWORD_LLM_MODEL", "google/flan-t5-base")
        print(f"[KeywordExtractor] Using LLM model: {model_name}")
        llm = pipeline(
            "text2text-generation",
            model=model_name,
            tokenizer=model_name,
            max_new_tokens=16,
            temperature=0.0
        )
        # Generate a prompt for the LLM to extract keywords
        prompt = (
            "Extract up to 3 concise research keywords (noun phrases) from the input. "
            "Only return comma-separated keywords.\n"
            f"Input: {user_input}"
        )
        # Get the LLM's output and parse it into a list of candidate keywords
        text = llm(prompt)[0]["generated_text"]
        candidates = [s.strip() for s in re.split(r"[;,]", text) if s.strip()]
        # Clean and filter the candidate keywords
        terms = _clean_terms(candidates)
        if terms:
            return " ".join(terms)
    except Exception as e:
        print(f"[KeywordExtractor] LLM extraction failed: {e}")

    # 2) KeyBERT fallback
    try:
        from keybert import KeyBERT
        # Initialize the KeyBERT model
        kw_model = KeyBERT()
        # Extract keywords using KeyBERT
        kws = kw_model.extract_keywords(
            user_input, keyphrase_ngram_range=(1, 3), stop_words="english", top_n=5
        )
        # Clean and filter the extracted keywords
        terms = _clean_terms([k for k, _ in kws])
        if terms:
            return " ".join(terms)
    except Exception as e:
        print(f"[KeywordExtractor] KeyBERT failed: {e}")

    # 3) Regex fallback
    # Extract words using a regex pattern and clean the results
    words = re.findall(r"[A-Za-z][A-Za-z0-9\-\./]*", user_input)
    terms = _clean_terms(words)
    # Return the cleaned terms or the original input if no terms were extracted
    return " ".join(terms) if terms else user_input.strip()