import streamlit as st
from langchain_core.messages import messages_from_dict, messages_to_dict

from app.services_factory import ServicesFactory
from app.ui.views.chat_view import render_chat_view
from core.services.chat_service import ChatService
from core.services.document_service import DocumentService
from infrastructure.ai.semantic_router import SemanticRouter
from infrastructure.files.loader import DocumentLoader
from infrastructure.llm.groq_provider import GroqProvider
from infrastructure.storage.session_manager import SessionManager
from infrastructure.vector_store.faiss_repository import FAISSRepository

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

# Estado de la sesión
if "session_id" not in st.session_state:
    st.session_state.session_id = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# --- Sidebar: Gestión de Espacios de Trabajo ---
with st.sidebar:
    st.title("🗂️ Espacios de Trabajo")
    
    if st.button("➕ Nueva Auditoría", use_container_width=True):
        st.session_state.session_id = None
        st.session_state.chat_history = []
        st.rerun()
    
    st.divider()
    st.subheader("Mis Proyectos")
    
    sessions = session_manager.list_sessions()
    for session in sessions:
        file_count = len(session.get('files', []))
        created_date = session.get('created_at', '')[:10]
        
        label = f"📂 {session['name']}\n\n📅 {created_date} | 📄 {file_count} docs"
        
        if st.button(label, key=session['id'], use_container_width=True):
            st.session_state.session_id = session['id']
            st.session_state.chat_history = [] # Limpiar visualmente antes de cargar
            st.rerun()

# --- Panel Principal ---

# MODO CREAR (Sin sesión seleccionada)
if st.session_state.session_id is None:
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.title("Iniciar Nuevo Análisis")
        st.markdown("### Plataforma de Auditoría Inteligente ISO 9001")
        st.markdown("Cree un espacio de trabajo dedicado para analizar normativas y procedimientos.")
        
        with st.form("create_session_form"):
            new_session_name = st.text_input("Nombre del Proyecto", placeholder="Ej: Auditoría Interna 2024")
            submitted = st.form_submit_button("Crear Espacio", type="primary", use_container_width=True)
            
            if submitted:
                if new_session_name.strip():
                    session_id = session_manager.create_session(new_session_name)
                    st.session_state.session_id = session_id
                    st.success(f"Proyecto '{new_session_name}' inicializado.")
                    st.rerun()
                else:
                    st.error("El nombre del proyecto es requerido.")

# MODO CHAT (Sesión seleccionada)
else:
    # Obtener datos de la sesión actual
    current_session = next((s for s in sessions if s['id'] == st.session_state.session_id), None)
    session_name = current_session['name'] if current_session else "Sesión Desconocida"
    session_path = session_manager.get_session_path(st.session_state.session_id)
    
    # Header con controles
    col_header, col_controls = st.columns([3, 1])
    
    with col_header:
        st.title(f"📑 {session_name}")
    
    with col_controls:
        # Popover para gestión de documentos
        with st.popover("📎 Gestionar Documentos", use_container_width=True):
            st.markdown("### Cargar Normativas")
            uploaded_files = st.file_uploader("Seleccionar archivos PDF", type=["pdf"], accept_multiple_files=True)
            
            if uploaded_files and st.button("Procesar e Indexar", type="primary", use_container_width=True):
                status_container = st.status("Iniciando proceso de ingestión...", expanded=True)
                try:
                    status_container.write("📄 Leyendo documentos PDF...")
                    # Simulación de pasos para UX (el proceso real ocurre en process_and_ingest_files)
                    
                    new_retriever, new_bm25, num_chunks = doc_service.process_and_ingest_files(
                        uploaded_files, session_path, vector_repo
                    )
                    
                    if num_chunks > 0:
                        status_container.write("🧠 Generando embeddings y actualizando índice vectorial...")
                        
                        # Inicializar ChatService con la DB actualizada
                        retriever, bm25 = vector_repo.get_vector_db(session_path)
                        
                        # Actualizar estado global si es necesario (aunque se hace abajo)
                        status_container.update(label="¡Indexación completada con éxito!", state="complete", expanded=False)
                        st.toast(f"✅ {num_chunks} fragmentos de conocimiento agregados.", icon="✅")
                    else:
                        status_container.update(label="Advertencia: No se extrajo texto.", state="error")
                        st.warning("No se pudo extraer texto de los archivos.")
                        
                except Exception as e:
                    status_container.update(label="Error en el proceso", state="error")
                    st.error(f"Error crítico: {str(e)}")

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

    # Persistencia del historial (Guardar estado actual)
    if st.session_state.chat_history:
        history_dicts = messages_to_dict(st.session_state.chat_history)
        session_manager.save_history(st.session_state.session_id, history_dicts)

    # Renderizar vista de chat
    render_chat_view(chat_service)

