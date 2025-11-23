import streamlit as st
import tempfile
from pathlib import Path
from langchain_core.messages import messages_from_dict, messages_to_dict

from app.services_factory import ServicesFactory
# Imports de componentes UI movidos a lazy loading para optimizar inicio
from core.services.chat_service import ChatService

# Configuración de la página - Estilo Profesional
st.set_page_config(page_title="Analista ISO 9001", layout="wide", page_icon="📑")

# Inicialización de componentes (Singleton pattern in session state)
if "components" not in st.session_state:
    st.session_state.components = ServicesFactory.create_services()

# --- ESTILOS CSS PERSONALIZADOS ---
def load_css():
    css_path = Path(__file__).parent / "app/ui/styles.css"
    if css_path.exists():
        with open(css_path) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()

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

# --- LÓGICA DE ENRUTAMIENTO PRINCIPAL ---

# 1. Si NO hay sesión seleccionada -> VISTA HOME
if not st.session_state.session_id:
    from app.ui.views.home_view import render_home_view
    render_home_view(session_manager)
    st.stop() # Detener ejecución aquí para no renderizar el resto

# 2. Si HAY sesión seleccionada -> VISTA DE PROYECTO
# Recuperar datos de sesión
session_path = session_manager.get_session_path(st.session_state.session_id)
session_name = session_manager.get_session_name(st.session_state.session_id)
is_draft = False # Ya no usamos modo borrador temporal, todo es proyecto

# Inicializar chat activo si no existe
if not st.session_state.active_chat_id:
    chats = session_manager.list_chats(st.session_state.session_id)
    if chats:
        st.session_state.active_chat_id = chats[0]['id']
    else:
        # Crear primer chat por defecto
        st.session_state.active_chat_id = session_manager.create_chat(st.session_state.session_id, "Chat Inicial")

# Inicialización del ChatService (Necesario para el Dashboard)
try:
    # --- OPTIMIZACIÓN: Caching de Retrievers en Session State ---
    cache_key = f"retrievers_{session_path}"
    
    if "active_retrievers_key" not in st.session_state or \
       st.session_state.active_retrievers_key != cache_key or \
       st.session_state.get("force_refresh_retrievers", False):
        
        with st.spinner("Cargando base de conocimientos..."):
            retriever, bm25 = vector_repo.get_vector_db(session_path)
            st.session_state.cached_retriever = retriever
            st.session_state.cached_bm25 = bm25
            st.session_state.active_retrievers_key = cache_key
            st.session_state.force_refresh_retrievers = False 
    
    chat_service = ChatService(llm_provider, vector_repo, doc_loader, router, prompt_manager)
    chat_service.vector_store = st.session_state.cached_retriever
    chat_service.bm25_retriever = st.session_state.cached_bm25
    
except Exception as e:
    st.error(f"Error inicializando servicios: {e}")
    st.stop()

# --- Sidebar: Gestión del Proyecto Activo ---
# Renderizar sidebar DESPUÉS de cargar datos para sincronizar la UI
from app.ui.components.sidebar import render_sidebar
render_sidebar(session_manager, doc_service, vector_repo, session_path, is_draft, chat_service=chat_service)

# Cargar Historial
if not st.session_state.chat_history and st.session_state.active_chat_id:
    history_dicts = session_manager.load_chat_history(st.session_state.session_id, st.session_state.active_chat_id)
    st.session_state.chat_history = messages_from_dict(history_dicts)

# Persistencia del historial
if st.session_state.chat_history and st.session_state.active_chat_id:
    history_dicts = messages_to_dict(st.session_state.chat_history)
    session_manager.save_chat_history(st.session_state.session_id, st.session_state.active_chat_id, history_dicts)

# Navegación Principal (Chat vs Cuestionarios)
if "current_view" not in st.session_state:
    st.session_state.current_view = "Chat"

if st.session_state.current_view == "Chat":
    from app.ui.views.chat_view import render_chat_view
    render_chat_view(chat_service, doc_service, vector_repo, feedback_logger, session_path)
elif st.session_state.current_view == "Cuestionarios":
    from app.ui.views.quiz_view import render_quiz_view
    render_quiz_view(chat_service)


