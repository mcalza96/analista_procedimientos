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
    
    # Rutas absolutas robustas
    PERSIST_DIRECTORY = str(BASE_DIR / "faiss_index")
    DOCSTORE_DIRECTORY = str(BASE_DIR / "docstore")
    TEMP_DOCS_DIR = str(BASE_DIR / "docs_temp")
    FEEDBACK_FILE = str(BASE_DIR / "feedback_log.csv")

settings = Settings()
