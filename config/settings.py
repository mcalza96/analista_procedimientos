import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    LLAMA_CLOUD_API_KEY = os.getenv("LLAMA_CLOUD_API_KEY")
    MODEL_NAME = "llama-3.3-70b-versatile"
    EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    PERSIST_DIRECTORY = "./faiss_index"
    TEMP_DOCS_DIR = "docs_temp"
    FEEDBACK_FILE = "feedback_log.csv"

settings = Settings()
