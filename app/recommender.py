from typing import List, Tuple, Dict
from langchain.docstore.document import Document
from app.vector_store import load_vector_db

def _normalize_title(title: str) -> str:
    """
    Normalize a paper title by converting it to lowercase, stripping whitespace,
    and collapsing multiple spaces into a single space.

    Args:
        title (str): The title to normalize.

    Returns:
        str: The normalized title.
    """
    return " ".join((title or "").strip().lower().split())

def recommend_similar_papers(query: str, k_papers: int = 5, fetch_k: int = 60) -> str:
    """
    Recommend related papers based on a query, ensuring strict de-duplication by title.
    Limits the output to a maximum of `k_papers` results.

    Args:
        query (str): The search query to find similar papers.
        k_papers (int): The maximum number of papers to recommend. Default is 5.
        fetch_k (int): The number of papers to fetch for similarity search. Default is 60.

    Returns:
        str: A formatted string containing the recommended papers or an error message.
    """
    try:
        # Load the vector database for similarity search
        db = load_vector_db()
    except Exception as e:
        # Return an error message if the vector database is unavailable
        return f"Vector DB not available: {e}"

    # Perform similarity search to retrieve many chunks
    try:
        results = db.similarity_search_with_score(query, k=fetch_k)
    except Exception:
        # Fallback to similarity search without scores if scoring fails
        docs = db.similarity_search(query, k=fetch_k)
        results = [(d, 0.0) for d in docs]

    seen_titles = set()  # Track seen titles to ensure de-duplication
    unique: List[Tuple[Document, float]] = []  # Store unique documents with scores

    for doc, score in results:
        # Extract and normalize the title from the document metadata
        title = (doc.metadata or {}).get("title") or "Untitled"
        tkey = _normalize_title(title)
        if tkey in seen_titles:
            # Skip duplicate titles
            continue
        seen_titles.add(tkey)
        unique.append((doc, score))
        if len(unique) >= k_papers:  # Stop if the desired number of papers is reached
            break

    if not unique:
        # Return a message if no related papers are found
        return "No related papers found."

    # Sort the unique papers by score (lower scores are better)
    unique.sort(key=lambda x: x[1])

    # Format the recommended papers into a readable string
    lines = []
    for doc, _ in unique[:k_papers]:  # Limit the output to `k_papers`
        m = doc.metadata or {}
        title = m.get("title") or "Untitled"
        authors = m.get("authors") or "Unknown authors"
        date = m.get("published") or "Unknown date"
        url = m.get("url") or m.get("abs_url") or m.get("pdf_url") or ""
        snippet = (doc.page_content or "").strip()
        if len(snippet) > 300:
            # Truncate the snippet if it exceeds 300 characters
            snippet = snippet[:300] + "..."
        lines.append(f"📄 {title}\n👤 {authors} | 📅 {date}\n{snippet}\n🔗 {url}")

    # Join the formatted lines with double line breaks and return
    return "\n\n".join(lines)