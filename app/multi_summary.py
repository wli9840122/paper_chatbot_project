from typing import List
from langchain.docstore.document import Document
from langchain.chains.summarize import load_summarize_chain
from app.llm_config import get_chat_model

def summarize_documents(docs: List[Document]) -> str:
    """
    Summarize a list of LangChain Document objects using the map-reduce summarization chain.

    Args:
        docs (List[Document]): A list of LangChain Document objects to summarize.

    Returns:
        str: A summary of the provided documents. If no documents are provided,
             returns a message indicating that there are no documents to summarize.
    """
    if not docs:
        # Return a message if the input document list is empty
        return "No documents to summarize."

    # Retrieve the chat model instance for performing the summarization
    llm = get_chat_model()

    # Load the map-reduce summarization chain with the chat model
    chain = load_summarize_chain(
        llm=llm,
        chain_type="map_reduce",
        verbose=False
    )

    # Run the summarization chain on the provided documents and return the result
    return chain.run(docs)