import streamlit as st
from typing import Any

def render_sidebar(session_manager: Any, doc_service: Any, vector_repo: Any, session_path: str, is_draft: bool, chat_service: Any = None):
    """
    Renderiza la barra lateral enfocada en el proyecto activo (Estilo NotebookLM).
    """
    with st.sidebar:
        # --- HEADER: VOLVER A HOME ---
        if st.button("⬅ Volver a Proyectos", use_container_width=True):
            st.session_state.session_id = None
            st.session_state.chat_history = []
            st.session_state.active_chat_id = None
            st.rerun()
            
        st.divider()
        
        # --- SECCIÓN 1: FUENTES (PRIORIDAD ALTA) ---
        st.subheader("📚 Fuentes")
        
        # Lista de archivos
        files_display = []
        if st.session_state.session_id:
            files_display = session_manager.get_session_files(st.session_state.session_id)
            
        if not files_display:
            st.caption("No hay fuentes añadidas.")
        else:
            for file in files_display:
                col_f_name, col_f_del = st.columns([5, 1])
                with col_f_name:
                    st.text(f"📄 {file}")
                with col_f_del:
                    if st.button("✕", key=f"del_file_{file}", help="Eliminar fuente"):
                        if doc_service.delete_file(session_path, file, vector_repo):
                            session_manager.remove_file_from_session(st.session_state.session_id, file)
                            st.session_state.force_refresh_retrievers = True
                            st.rerun()

        # Añadir Fuente (Expander)
        with st.expander("➕ Añadir Fuente"):
            uploaded_files = st.file_uploader(
                "Subir archivos",
                type=["pdf", "docx", "txt", "pptx", "xlsx"],
                accept_multiple_files=True,
                label_visibility="collapsed"
            )
            
            if uploaded_files:
                if st.button("Procesar", type="primary", use_container_width=True):
                    with st.spinner("Procesando..."):
                        try:
                            new_retriever, new_bm25, num_chunks = doc_service.process_and_ingest_files(
                                uploaded_files, session_path, vector_repo
                            )
                            if num_chunks > 0:
                                filenames = [f.name for f in uploaded_files]
                                session_manager.add_files_to_session(st.session_state.session_id, filenames)
                                st.session_state.force_refresh_retrievers = True
                                
                                # Generar Resumen Automático si tenemos chat_service
                                if chat_service and not is_draft:
                                    with st.spinner("Generando resumen del proyecto..."):
                                        # Actualizar el servicio con el nuevo retriever INMEDIATAMENTE
                                        # para que el resumen vea los nuevos documentos
                                        if new_retriever:
                                            chat_service.vector_store = new_retriever
                                        if new_bm25:
                                            chat_service.bm25_retriever = new_bm25
                                            
                                        summary = chat_service.generate_context_summary()
                                        session_manager.update_session_summary(st.session_state.session_id, summary)
                                
                                st.rerun()
                        except Exception as e:
                            st.error(f"Error: {e}")

        st.divider()

        # --- SECCIÓN 2: NAVEGACIÓN DE VISTAS ---
        st.subheader("Herramientas")
        view = st.radio(
            "Ir a:", 
            ["Chat", "Cuestionarios"], 
            index=0 if st.session_state.get("current_view", "Chat") == "Chat" else 1,
            label_visibility="collapsed"
        )
        st.session_state.current_view = view

        st.divider()

        # --- SECCIÓN 3: CHATS GUARDADOS ---
        st.subheader("💬 Chats")
        
        if st.button("➕ Nuevo Chat", use_container_width=True):
            new_chat_id = session_manager.create_chat(st.session_state.session_id)
            st.session_state.active_chat_id = new_chat_id
            st.session_state.chat_history = []
            st.rerun()
            
        chats = session_manager.list_chats(st.session_state.session_id)
        
        for chat in chats:
            c_col1, c_col2 = st.columns([5, 1])
            is_active = (chat['id'] == st.session_state.active_chat_id)
            
            # Estilo visual para chat activo
            label = f"{'🟢' if is_active else '⚪'} {chat['title']}"
            
            with c_col1:
                if st.button(label, key=f"chat_{chat['id']}", use_container_width=True):
                    st.session_state.active_chat_id = chat['id']
                    st.session_state.chat_history = [] # Forzar recarga
                    st.rerun()
                    
            with c_col2:
                if st.button("✕", key=f"del_chat_{chat['id']}"):
                    if session_manager.delete_chat(st.session_state.session_id, chat['id']):
                        if is_active:
                            st.session_state.active_chat_id = None
                            st.session_state.chat_history = []
                        st.rerun()
