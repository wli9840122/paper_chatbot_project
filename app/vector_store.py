import os
from typing import List
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from app.config import CHUNK_SIZE, CHUNK_OVERLAP, VECTOR_DIR

LOCAL_EMBEDDING_PATH = os.getenv("LOCAL_EMBEDDING_PATH", "")  # Path to the local embedding model
EMBEDDING_MODEL_ONLINE = os.getenv("EMBEDDING_MODEL_ONLINE", "sentence-transformers/all-MiniLM-L6-v2")  # Online embedding model name
ALLOW_ONLINE_FALLBACK = os.getenv("ALLOW_ONLINE_FALLBACK", "false").lower() == "true"  # Flag to allow fallback to online embedding model


def _load_embedding_model():
    """
    Load embeddings model with preference for local files.
    Falls back to Hugging Face online model if allowed.

    Returns:
        HuggingFaceEmbeddings: The embedding model instance.

    Raises:
        FileNotFoundError: If the local embedding model is not found and online fallback is disabled.
    """
    if LOCAL_EMBEDDING_PATH and os.path.exists(LOCAL_EMBEDDING_PATH):
        # Use the local embedding model if it exists
        model_name_or_path = LOCAL_EMBEDDING_PATH
    else:
        if not ALLOW_ONLINE_FALLBACK:
            # Raise an error if the local model is not found and fallback is disabled
            raise FileNotFoundError(
                f"Local embedding model not found at {LOCAL_EMBEDDING_PATH} "
                "and online fallback is disabled."
            )
        # Use the online embedding model as a fallback
        model_name_or_path = EMBEDDING_MODEL_ONLINE

    # Return the embedding model instance
    return HuggingFaceEmbeddings(
        model_name=model_name_or_path,
        model_kwargs={"local_files_only": not ALLOW_ONLINE_FALLBACK}
    )


embedding_model = _load_embedding_model()  # Load the embedding model


def _split_docs(docs: List[Document]) -> List[Document]:
    """
    Split documents into smaller chunks using a recursive character text splitter.

    Args:
        docs (List[Document]): A list of documents to split.

    Returns:
        List[Document]: A list of smaller document chunks.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,  # Maximum size of each chunk
        chunk_overlap=CHUNK_OVERLAP,  # Overlap size between chunks
        separators=["\n\n", "\n", " ", ""],  # Separators to use for splitting
    )
    return splitter.split_documents(docs)  # Split the documents and return the chunks


def build_vector_db_from_texts(texts: List[str], save_path: str = VECTOR_DIR):
    """
    Build a FAISS vector store from raw texts, automatically chunking them before embedding.

    Args:
        texts (List[str]): A list of raw text strings to index.
        save_path (str): The directory path to save the vector store. Defaults to VECTOR_DIR.
    """
    raw_docs = [Document(page_content=t) for t in texts]  # Convert raw texts to Document objects
    chunks = _split_docs(raw_docs)  # Split the documents into chunks
    vectorstore = FAISS.from_documents(chunks, embedding_model)  # Create a FAISS vector store from the chunks
    os.makedirs(save_path, exist_ok=True)  # Ensure the save directory exists
    vectorstore.save_local(save_path)  # Save the vector store to the specified path


def build_vector_db_from_documents(docs: List[Document], save_path: str = VECTOR_DIR):
    """
    Build a FAISS vector store from Document objects, automatically chunking them before embedding.

    Args:
        docs (List[Document]): A list of Document objects to index.
        save_path (str): The directory path to save the vector store. Defaults to VECTOR_DIR.
    """
    chunks = _split_docs(docs)  # Split the documents into chunks
    vectorstore = FAISS.from_documents(chunks, embedding_model)  # Create a FAISS vector store from the chunks
    os.makedirs(save_path, exist_ok=True)  # Ensure the save directory exists
    vectorstore.save_local(save_path)  # Save the vector store to the specified path


def load_vector_db(path: str = VECTOR_DIR) -> FAISS:
    """
    Load a FAISS vector store from a local directory.

    Args:
        path (str): The directory path to load the vector store from. Defaults to VECTOR_DIR.

    Returns:
        FAISS: The loaded FAISS vector store instance.

    Raises:
        ValueError: If the specified directory does not exist.
    """
    if not os.path.exists(path):
        # Raise an error if the directory does not exist
        raise ValueError("Vector DB directory not found: " + path)
    # Load and return the FAISS vector store
    return FAISS.load_local(path, embedding_model, allow_dangerous_deserialization=True)