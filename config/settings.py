import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Definir ruta raíz dinámica
BASE_DIR = Path(__file__).resolve().parent.parent

class Settings:
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    LLAMA_CLOUD_API_KEY = os.getenv("LLAMA_CLOUD_API_KEY")
    
    MODEL_NAME = "llama-3.3-70b-versatile"
    EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    RERANKER_MODEL = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
    
    # RAG Configuration
    CHUNK_SIZE_CHILD = 400
    CHUNK_OVERLAP_CHILD = 50
    CHUNK_SIZE_PARENT = 2000
    CHUNK_OVERLAP_PARENT = 200
    RETRIEVER_K_PARENT = 60
    RETRIEVER_K_BM25 = 30
    RERANKER_TOP_K = 5
    
    # LLM Configuration
    LLM_TEMPERATURE = 0.1
    
    # Rutas absolutas robustas
    FEEDBACK_FILE = str(BASE_DIR / "feedback_log.csv")

settings = Settings()
