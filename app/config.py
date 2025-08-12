import os
from dotenv import load_dotenv
load_dotenv()

USE_OPENAI = os.getenv("USE_OPENAI", "false").lower() == "true"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# Centralized knobs for chunking & retrieval

# Text chunking for ingest & summarization
CHUNK_SIZE = 1200          # ~800–1200 tokens worth of chars if using char splitter
CHUNK_OVERLAP = 150

# Retrieval behavior
RETRIEVAL_TOP_K = 5        # final context chunks sent to LLM
RETRIEVAL_FETCH_K = 20     # initial pool for MMR to de-duplicate

USE_HYBRID=True
BM25_K=8
VECTOR_K=8
HYBRID_WEIGHTS=0.4,0.6
USE_LLM_KEYWORD=True

FULL_TEXT_INGEST=False   # true if switch to full PDF ingestion

REDIS_URL="redis://localhost:6379/0"

SESSION_FILE=".chat_sessions.json"

VECTOR_DIR="data/papers/"