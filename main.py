import streamlit as st
import os

# Fix Tokenizers Parallelism Warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from config.settings import settings
from infrastructure.llm.groq_provider import GroqProvider
from infrastructure.vector_store.faiss_repository import FAISSRepository
from core.services.chat_service import ChatService
from core.services.quiz_service import QuizService
from app.ui.views.chat_view import render_chat_view
from app.ui.views.quiz_view import render_quiz_view

# Page Config
st.set_page_config(page_title="Asistente ISO 9001", page_icon="🧪", layout="wide")

# Dependency Injection
@st.cache_resource
def get_services():
    llm_provider = GroqProvider()
    vector_store_repo = FAISSRepository()
    chat_service = ChatService(llm_provider, vector_store_repo)
    quiz_service = QuizService(llm_provider, chat_service)
    
    # Load existing DB if available
    chat_service.load_existing_db()
    
    return chat_service, quiz_service

try:
    chat_service, quiz_service = get_services()
except Exception as e:
    st.error(f"Error initializing services: {e}")
    st.stop()

# Sidebar Navigation
with st.sidebar:
    st.title("🧪 ISO Lab Assistant")
    st.markdown("---")
    mode = st.radio("Modo", ["💬 Chat Asistente", "🎓 Entrenador (Quiz)"])
    st.markdown("---")
    
    # File Upload (Global Context)
    st.subheader("📂 Base de Conocimiento")
    uploaded_files = st.file_uploader(
        "Subir manuales PDF", 
        type=["pdf"], 
        accept_multiple_files=True
    )
    
    if uploaded_files:
        if st.button("Procesar Documentos"):
            with st.spinner("Procesando documentos..."):
                # Save files
                if not os.path.exists(settings.TEMP_DOCS_DIR):
                    os.makedirs(settings.TEMP_DOCS_DIR)
                
                file_paths = []
                for uploaded_file in uploaded_files:
                    file_path = os.path.join(settings.TEMP_DOCS_DIR, uploaded_file.name)
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    file_paths.append(file_path)
                
                # Process
                chat_service.process_uploaded_files(file_paths)
                st.success("Documentos procesados y listos.")

# Main Content
if mode == "💬 Chat Asistente":
    render_chat_view(chat_service)
elif mode == "🎓 Entrenador (Quiz)":
    render_quiz_view(quiz_service)
