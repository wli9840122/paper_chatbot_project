import os
from typing import Dict
from dotenv import load_dotenv
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
# from langchain_community.llms import HuggingFacePipeline
from langchain_community.chat_message_histories import RedisChatMessageHistory
# from langchain_community.chat_models import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.messages import SystemMessage, HumanMessage

# from transformers import pipeline
from app.retriever_factory import get_retriever
from app.llm_config import get_chat_model
from app.config import VECTOR_DIR
from typing import List

load_dotenv()

# ===== Config =====
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

# ===== Session storage for QA =====
qa_sessions: Dict[str, ConversationalRetrievalChain] = {}

# --------------------------
# Helpers
# --------------------------

def format_messages_as_prompt(messages):
    """
    Convert a list of LangChain message objects into a single string prompt.

    Args:
        messages (List[Message]): A list of message objects.

    Returns:
        str: A single string containing the concatenated message contents.
    """
    return "\n".join(m.content for m in messages)

def get_memory(chat_id: str):
    """
    Retrieve a persistent memory object for a given chat session ID.
    Uses Redis to store and retrieve conversation history.

    Args:
        chat_id (str): The unique identifier for the chat session.

    Returns:
        ConversationBufferMemory: A memory object for storing chat history.
    """
    history = RedisChatMessageHistory(session_id=chat_id, url=REDIS_URL)
    return ConversationBufferMemory(chat_memory=history, return_messages=True)

def load_vector_db(path: str = VECTOR_DIR):
    """
    Load a vector database from the specified path.

    Args:
        path (str): The file path to the vector database.

    Returns:
        FAISS: The loaded FAISS vector database.
    """
    embedding_model = HuggingFaceEmbeddings()
    return FAISS.load_local(path, embedding_model, allow_dangerous_deserialization=True)

# ===== Multi-turn QA =====
def ask_question_with_rag(question: str, context_docs: List[str], chat_id: str) -> str:
    """
    Answer a user's question using a Retrieval-Augmented Generation (RAG) approach.

    Args:
        question (str): The user's question.
        context_docs (List[str]): A list of context documents to base the answer on.
        chat_id (str): The unique identifier for the chat session.

    Returns:
        str: The generated answer to the user's question.
    """
    llm = get_chat_model()  # Retrieve the chat model instance.

    memory = ConversationBufferMemory(return_messages=True)
    context_text = "\n\n".join(context_docs)
    system_prompt = (
        "You are a professional AI paper assistant. "
        "Answer the user's question strictly based on the provided paper content. "
        "Do not fabricate information.\n\n"
        f"Context:\n{context_text}"
    )
    history_messages = memory.chat_memory.messages
    full_prompt = format_messages_as_prompt(
        [SystemMessage(content=system_prompt)] + history_messages + [HumanMessage(content=question)]
    )

    response = llm.invoke(full_prompt)
    answer = response.content if hasattr(response, "content") else str(response)

    memory.chat_memory.add_user_message(question)
    memory.chat_memory.add_ai_message(answer)
    return answer