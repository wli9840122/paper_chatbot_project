from typing import List
from datetime import datetime
from langchain.docstore.document import Document

# Uses the lightweight 'arxiv' package. Add to requirements: arxiv>=2.1.0
import arxiv


def _fmt_date(dt: datetime | None) -> str:
    """
    Format a datetime object into a string in the format 'YYYY-MM-DD'.

    Args:
        dt (datetime | None): The datetime object to format.

    Returns:
        str: The formatted date string or 'Unknown date' if the input is None.
    """
    if not dt:
        return "Unknown date"
    return dt.strftime("%Y-%m-%d")


def search_arxiv(query: str, max_results: int = 5) -> List[Document]:
    """
    Search ArXiv and return a list of LangChain Documents with rich metadata.

    Args:
        query (str): The search query string.
        max_results (int): The maximum number of results to retrieve. Defaults to 5.

    Returns:
        List[Document]: A list of LangChain Document objects containing:
            - title, authors (comma-joined), published (YYYY-MM-DD),
              url (pdf link), entry_id (abs page), primary_category.
            The summary is stored in the page_content.
    """
    client = arxiv.Client()  # Initialize the ArXiv client with default rate limits
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance,  # Sort results by relevance
    )

    docs: List[Document] = []
    for result in client.results(search):
        # Extract metadata from the ArXiv result
        title = result.title or "Untitled"
        authors = ", ".join(a.name for a in result.authors) or "Unknown authors"
        published = _fmt_date(result.published)
        pdf_url = result.pdf_url or ""
        abs_url = result.entry_id or ""
        summary = (result.summary or "").strip()

        metadata = {
            "title": title,
            "authors": authors,
            "published": published,
            "url": pdf_url or abs_url,
            "pdf_url": pdf_url,
            "abs_url": abs_url,
            "primary_category": getattr(result, "primary_category", "") or "",
            "arxiv_id": result.get_short_id() if hasattr(result, "get_short_id") else "",
        }

        # Create a LangChain Document object with the summary and metadata
        docs.append(Document(page_content=summary, metadata=metadata))

    return docs