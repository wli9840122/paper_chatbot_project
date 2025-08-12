from langchain.docstore.document import Document

def make_doc(title="Test Paper", authors="Doe, J.", date="2025-01-01", url="http://x", text="abstract"):
    return Document(page_content=text, metadata={
        "title": title, "authors": authors, "published": date, "url": url, "arxiv_id": ""
    })