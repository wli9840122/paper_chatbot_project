from typing import List
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from app.vector_store import load_vector_db, build_vector_db_from_texts
from app.retriever_factory import refresh_bm25_cache
from app.arxiv_search import search_arxiv

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100


def _chunk_docs(docs: List[Document]) -> List[Document]:
    """
    Split long documents into smaller chunks using a recursive character text splitter.

    Args:
        docs (List[Document]): A list of LangChain Document objects to be split.

    Returns:
        List[Document]: A list of smaller chunked Document objects.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""],
    )
    return splitter.split_documents(docs)


def ingest_arxiv_to_vector_db(query: str, full_text: bool = False):
    """
    Search ArXiv for documents, filter duplicates, optionally chunk documents,
    store them in the vector database, and refresh the BM25 retriever cache.

    Args:
        query (str): The search keyword or phrase to query ArXiv.
        full_text (bool): If True, assumes documents contain full paper text
                          and splits them into smaller chunks.

    Returns:
        int: The number of new documents added to the vector database.
    """
    print(f"[Ingest] Searching ArXiv for: {query}")
    docs = search_arxiv(query)
    if not docs:
        print("[Ingest] No documents found.")
        return 0

    try:
        # Load the existing vector database
        db = load_vector_db()
    except Exception:
        db = None

    # Avoid duplicates by checking existing document titles
    existing_titles = set()
    if db:
        try:
            existing_docs = db.similarity_search(query, k=100)
            existing_titles = {getattr(d, "metadata", {}).get("title") for d in existing_docs}
        except Exception as e:
            print(f"[Ingest] Could not fetch existing titles: {e}")

    # Filter out documents with titles already in the database
    new_docs = [d for d in docs if d.metadata.get("title") not in existing_titles]

    if not new_docs:
        print("[Ingest] No new documents to add.")
        return 0

    # If full_text mode is enabled, split documents into smaller chunks
    if full_text:
        print("[Ingest] Chunking full-text documents...")
        new_docs = _chunk_docs(new_docs)

    # Add the new documents to the vector database
    print(f"[Ingest] Adding {len(new_docs)} new documents to vector DB.")
    build_vector_db_from_texts([doc.page_content for doc in new_docs])

    # Refresh the BM25 retriever cache
    refresh_bm25_cache()
    print("[Ingest] BM25 retriever refreshed.")

    return len(new_docs)