import streamlit as st
import tempfile
from pathlib import Path
from langchain_core.messages import messages_from_dict, messages_to_dict

from app.services_factory import ServicesFactory
from app.ui.views.chat_view import render_chat_view
from app.ui.components.sidebar import render_sidebar
from app.ui.components.dashboard import render_dashboard
from core.services.chat_service import ChatService

# Configuración de la página - Estilo Profesional
st.set_page_config(page_title="Analista ISO 9001", layout="wide", page_icon="📑")

# Inicialización de componentes (Singleton pattern in session state)
if "components" not in st.session_state:
    st.session_state.components = ServicesFactory.create_services()

# Referencias rápidas
session_manager = st.session_state.components["session_manager"]
llm_provider = st.session_state.components["llm_provider"]
vector_repo = st.session_state.components["vector_repo"]
doc_loader = st.session_state.components["doc_loader"]
router = st.session_state.components["router_repo"]
doc_service = st.session_state.components["doc_service"]
feedback_logger = st.session_state.components["feedback_logger"]
prompt_manager = st.session_state.components["prompt_manager"]

# Estado de la sesión
if "session_id" not in st.session_state:
    st.session_state.session_id = None
if "active_chat_id" not in st.session_state:
    st.session_state.active_chat_id = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "temp_session_path" not in st.session_state:
    # Crear directorio temporal para modo borrador
    st.session_state.temp_session_path = tempfile.mkdtemp()
sessions = session_manager.list_sessions()

# Determinar ruta de sesión (Draft vs Saved) ANTES del Sidebar
if st.session_state.session_id:
    session_path = session_manager.get_session_path(st.session_state.session_id)
    current_session = next((s for s in sessions if s['id'] == st.session_state.session_id), None)
    session_name = current_session['name'] if current_session else "Sesión Desconocida"
    is_draft = False
    
    # Inicializar chat activo si no existe
    if not st.session_state.active_chat_id:
        chats = session_manager.list_chats(st.session_state.session_id)
        if chats:
            st.session_state.active_chat_id = chats[0]['id']
        else:
            # Crear primer chat por defecto
            st.session_state.active_chat_id = session_manager.create_chat(st.session_state.session_id, "Chat Inicial")
else:
    session_path = st.session_state.temp_session_path
    session_name = "Borrador (No Guardado)"
    is_draft = True
    st.session_state.active_chat_id = None # Borrador usa historial en memoria solamente

# --- Sidebar: Gestión de Espacios de Trabajo ---
render_sidebar(session_manager, doc_service, vector_repo, session_path, is_draft)

# Inicialización del ChatService (Necesario para el Dashboard)
try:
    # Asegurar directorios en temp si no existen (para FAISS)
    if is_draft:
        (Path(session_path) / "vector_store").mkdir(exist_ok=True)
        (Path(session_path) / "doc_store").mkdir(exist_ok=True)

    retriever, bm25 = vector_repo.get_vector_db(session_path)
    
    chat_service = ChatService(llm_provider, vector_repo, doc_loader, router, prompt_manager)
    chat_service.vector_store = retriever
    chat_service.bm25_retriever = bm25
    
except Exception as e:
    st.error(f"Error inicializando servicios: {e}")
    st.stop()

# --- Panel Principal ---
render_dashboard(session_manager, chat_service, session_name, session_path, is_draft)

# Cargar Historial (Solo si es sesión guardada y está vacío en estado)
if not is_draft and not st.session_state.chat_history and st.session_state.active_chat_id:
    history_dicts = session_manager.load_chat_history(st.session_state.session_id, st.session_state.active_chat_id)
    st.session_state.chat_history = messages_from_dict(history_dicts)

# Persistencia del historial (Solo si es sesión guardada)
if not is_draft and st.session_state.chat_history and st.session_state.active_chat_id:
    history_dicts = messages_to_dict(st.session_state.chat_history)
    session_manager.save_chat_history(st.session_state.session_id, st.session_state.active_chat_id, history_dicts)

# Renderizar vista de chat
render_chat_view(chat_service, doc_service, vector_repo, feedback_logger, session_path)


