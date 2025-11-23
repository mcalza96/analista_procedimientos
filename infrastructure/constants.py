from pathlib import Path

# Directory Names
DIR_SESSIONS = "sessions"
DIR_VECTOR_STORE = "vector_store"
DIR_DOC_STORE = "doc_store"
DIR_CHATS = "chats"
DIR_RAW_FILES = "raw_files"

# File Names
FILE_METADATA = "metadata.json"
FILE_HISTORY_LEGACY = "history.json"
FILE_FAISS_INDEX = "index.faiss"

# CSV Headers
FEEDBACK_HEADERS = ["Timestamp", "Pregunta", "Respuesta", "Calificación", "Detalle"]

# Formats
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Router Categories
ROUTER_CATEGORIES = ["PRECISION", "ANALYSIS", "CHAT"]

# Default Values
DEFAULT_SESSION_NAME = "Sesión Desconocida"
DEFAULT_CHAT_TITLE_PREFIX = "Chat"
