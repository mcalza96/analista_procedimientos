import streamlit as st
import os
import shutil
from langchain_core.messages import messages_to_dict, messages_from_dict

# Fix Tokenizers Parallelism Warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from infrastructure.storage.session_manager import SessionManager
from infrastructure.llm.groq_provider import GroqProvider
from infrastructure.vector_store.faiss_repository import FAISSRepository
from infrastructure.files.loader import DocumentLoader
from infrastructure.ai.semantic_router import SemanticRouter
from core.services.chat_service import ChatService
from app.ui.views.chat_view import render_chat_view

# Configuración de la página
st.set_page_config(page_title="Gestor de Conocimiento ISO 9001", layout="wide", page_icon="🍳")

# Inicialización de componentes (Singleton pattern in session state)
if "components" not in st.session_state:
    st.session_state.components = {
        "session_manager": SessionManager(),
        "llm_provider": GroqProvider(),
        "vector_repo": FAISSRepository(),
        "doc_loader": DocumentLoader(),
        "router": SemanticRouter()
    }

# Referencias rápidas
session_manager = st.session_state.components["session_manager"]
llm_provider = st.session_state.components["llm_provider"]
vector_repo = st.session_state.components["vector_repo"]
doc_loader = st.session_state.components["doc_loader"]
router = st.session_state.components["router"]

# Estado de la sesión
if "session_id" not in st.session_state:
    st.session_state.session_id = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# --- Sidebar: Gestión de Sesiones ---
with st.sidebar:
    st.title("🍳 Cocinas Aisladas")
    
    if st.button("➕ Nueva Cocina", use_container_width=True):
        st.session_state.session_id = None
        st.session_state.chat_history = []
        st.rerun()
    
    st.divider()
    st.subheader("Mis Cocinas")
    
    sessions = session_manager.list_sessions()
    for session in sessions:
        # Usamos un botón para seleccionar la sesión
        if st.button(f"📁 {session['name']}", key=session['id'], use_container_width=True):
            st.session_state.session_id = session['id']
            st.session_state.chat_history = [] # Limpiar visualmente antes de cargar
            st.rerun()

# --- Panel Principal ---

# MODO CREAR (Sin sesión seleccionada)
if st.session_state.session_id is None:
    st.title("🚀 Crear Nueva Cocina de Conocimiento")
    st.markdown("Define un nuevo espacio de trabajo aislado para tus documentos.")
    
    with st.form("create_session_form"):
        new_session_name = st.text_input("Nombre de la Cocina", placeholder="Ej: Auditoría 2024")
        submitted = st.form_submit_button("Crear Cocina")
        
        if submitted:
            if new_session_name.strip():
                session_id = session_manager.create_session(new_session_name)
                st.session_state.session_id = session_id
                st.success(f"Cocina '{new_session_name}' creada.")
                st.rerun()
            else:
                st.error("El nombre no puede estar vacío.")

# MODO CHAT (Sesión seleccionada)
else:
    # Obtener datos de la sesión actual
    current_session = next((s for s in sessions if s['id'] == st.session_state.session_id), None)
    session_name = current_session['name'] if current_session else "Sesión Desconocida"
    session_path = session_manager.get_session_path(st.session_state.session_id)
    
    st.title(f"🍳 {session_name}")
    
    # Cargar Historial
    if not st.session_state.chat_history:
        history_dicts = session_manager.load_history(st.session_state.session_id)
        st.session_state.chat_history = messages_from_dict(history_dicts)

    # Inicializar ChatService con la DB de esta sesión
    try:
        retriever, bm25 = vector_repo.get_vector_db(session_path)
        
        chat_service = ChatService(llm_provider, vector_repo, doc_loader, router)
        chat_service.vector_store = retriever
        chat_service.bm25_retriever = bm25
        
    except Exception as e:
        st.error(f"Error cargando la base de datos de la sesión: {e}")
        st.stop()

    # Zona de Carga de Documentos (Expander)
    with st.expander("📥 Agregar documentos a esta cocina"):
        uploaded_files = st.file_uploader("Subir PDFs", type=["pdf"], accept_multiple_files=True)
        
        if uploaded_files and st.button("Procesar e Integrar"):
            with st.spinner("Procesando ingredientes..."):
                # 1. Guardar temporalmente
                temp_dir = os.path.join(session_path, "temp_uploads")
                os.makedirs(temp_dir, exist_ok=True)
                
                file_paths = []
                for uploaded_file in uploaded_files:
                    file_path = os.path.join(temp_dir, uploaded_file.name)
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    file_paths.append(file_path)
                
                # 2. Cargar Documentos
                chunks = doc_loader.load_documents(file_paths)
                
                # 3. Agregar a la DB
                if chunks:
                    # Actualizamos los retrievers del servicio actual
                    new_retriever, new_bm25 = vector_repo.add_documents(session_path, chunks)
                    chat_service.vector_store = new_retriever
                    chat_service.bm25_retriever = new_bm25
                    st.success(f"✅ {len(chunks)} fragmentos agregados al conocimiento.")
                else:
                    st.warning("No se pudo extraer texto de los archivos.")
                
                # 4. Limpieza
                shutil.rmtree(temp_dir)

    # Persistencia del historial (Guardar estado actual)
    if st.session_state.chat_history:
        history_dicts = messages_to_dict(st.session_state.chat_history)
        session_manager.save_history(st.session_state.session_id, history_dicts)

    # Renderizar vista de chat
    render_chat_view(chat_service)
