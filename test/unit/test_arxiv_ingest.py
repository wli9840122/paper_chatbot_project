# app/ingest_arxiv.py
from typing import Optional, List
from langchain.docstore.document import Document
from app.arxiv_search import search_arxiv
from app.vector_store import build_vector_db_from_documents
import os

def ingest_arxiv_to_vector_db(
    keyword: str,
    max_results: int = 20,
    category: Optional[str] = None,
    year_from: Optional[int] = None,
    vector_dir: Optional[str] = None,
    dedup: bool = True,
) -> int:
    """
    Search ArXiv and ingest results into the FAISS vector store.
    Returns the number of documents ingested.
    """
    docs: List[Document] = search_arxiv(
        keyword,
        max_results=max_results,          # <-- pass it through
        category=category,
        year_from=year_from,
    )
    if not docs:
        return 0

    # optional de-dup by arxiv_id or title
    if dedup:
        seen = set()
        unique = []
        for d in docs:
            key = d.metadata.get("arxiv_id") or d.metadata.get("title")
            if key and key not in seen:
                seen.add(key)
                unique.append(d)
        docs = unique

    save_path = vector_dir or os.getenv("VECTOR_DIR", "data/papers")
    os.makedirs(save_path, exist_ok=True)
    build_vector_db_from_documents(docs, save_path=save_path)
    return len(docs)
