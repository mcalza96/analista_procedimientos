import streamlit as st
import tempfile
from typing import Any

def render_sidebar(session_manager: Any, doc_service: Any, vector_repo: Any, session_path: str, is_draft: bool):
    """
    Renderiza la barra lateral con la gestión de espacios de trabajo y base de conocimiento.
    """
    with st.sidebar:
        st.title("🗂️ Espacios de Trabajo")
        
        if st.button("➕ Nueva Auditoría", use_container_width=True):
            st.session_state.session_id = None
            st.session_state.chat_history = []
            # Crear nuevo temp path para limpiar el anterior
            st.session_state.temp_session_path = tempfile.mkdtemp()
            st.rerun()
        
        # --- Knowledge Manager Unificado ---
        st.divider()
        st.subheader("📂 Base de Conocimiento")
        
        # 1. ZONA DE CARGA (INPUT)
        with st.expander("📤 Cargar Documentos", expanded=True):
            uploaded_files = st.file_uploader(
                "Añadir normativas o evidencias",
                type=["pdf", "docx", "txt", "pptx", "xlsx"],
                accept_multiple_files=True
            )
            
            if uploaded_files:
                if st.button("Procesar Archivos", type="primary", use_container_width=True):
                    with st.status("Analizando documentos...", expanded=True) as status:
                        st.write("📥 Leyendo archivos...")
                        # Guardar y procesar
                        try:
                            new_retriever, new_bm25, num_chunks = doc_service.process_and_ingest_files(
                                uploaded_files, session_path, vector_repo
                            )
                            
                            if num_chunks > 0:
                                st.write("🧠 Generando embeddings...")
                                st.write("💾 Guardando en memoria vectorial...")
                                
                                # Actualizar lista de archivos en metadata si es sesión guardada
                                if not is_draft:
                                    filenames = [f.name for f in uploaded_files]
                                    session_manager.add_files_to_session(st.session_state.session_id, filenames)
                                
                                # Flag para actualizar resumen automáticamente
                                st.session_state.needs_summary_update = True
                                    
                                status.update(label="¡Procesamiento completado!", state="complete", expanded=False)
                                st.rerun()
                            else:
                                status.update(label="Advertencia: No se extrajo texto.", state="error")
                        except Exception as e:
                            status.update(label="Error en el proceso", state="error")
                            st.error(f"Error crítico: {str(e)}")

        # 2. ZONA DE GESTIÓN (LIST & DELETE)
        st.caption(f"ARCHIVOS EN MEMORIA")
        
        # Obtener lista de archivos: Preferir metadatos de sesión, fallback a disco
        files_display = []
        files_on_disk = doc_service.list_files(session_path)
        
        if st.session_state.session_id:
            # Sesión guardada: Usar metadatos como fuente de verdad
            files_display = session_manager.get_session_files(st.session_state.session_id)
            # Si metadatos está vacío pero hay archivos en disco (caso legacy o desincronizado), agregarlos
            for f in files_on_disk:
                if f not in files_display:
                    files_display.append(f)
        else:
            # Borrador: Usar disco
            files_display = files_on_disk

        if not files_display:
            st.info("No hay documentos indexados.")
        else:
            for file in files_display:
                col_f_name, col_f_del = st.columns([4, 1])
                
                # Verificar estado físico
                is_missing = file not in files_on_disk
                
                with col_f_name:
                    if is_missing:
                        st.markdown(f"📄 {file} <span style='color:red; font-size:0.8em'>(Fuente perdida)</span>", unsafe_allow_html=True)
                    else:
                        st.text(f"📄 {file}")
                
                with col_f_del:
                    if st.button("🗑️", key=f"del_file_{file}", help="Eliminar archivo y reindexar"):
                        with st.spinner("Reconstruyendo base de conocimiento..."):
                            # Intentar borrar (doc_service ahora maneja archivos faltantes)
                            if doc_service.delete_file(session_path, file, vector_repo):
                                # Actualizar metadatos si es sesión guardada
                                if st.session_state.session_id:
                                    session_manager.remove_file_from_session(st.session_state.session_id, file)
                                    
                                st.success("Eliminado")
                                st.rerun()
                            else:
                                st.error("Error eliminando archivo.")

        # --- GESTIÓN DE CHATS (Solo si hay sesión activa) ---
        if not is_draft and st.session_state.session_id:
            st.divider()
            st.subheader("💬 Chats del Proyecto")
            
            if st.button("➕ Nuevo Chat", use_container_width=True):
                new_chat_id = session_manager.create_chat(st.session_state.session_id)
                st.session_state.active_chat_id = new_chat_id
                st.session_state.chat_history = []
                st.rerun()
                
            chats = session_manager.list_chats(st.session_state.session_id)
            
            for chat in chats:
                c_col1, c_col2 = st.columns([4, 1])
                is_active = (chat['id'] == st.session_state.active_chat_id)
                
                # Estilo visual para chat activo
                label = f"{'🟢' if is_active else '⚪'} {chat['title']}"
                
                with c_col1:
                    if st.button(label, key=f"chat_{chat['id']}", use_container_width=True):
                        st.session_state.active_chat_id = chat['id']
                        st.session_state.chat_history = [] # Forzar recarga
                        st.rerun()
                        
                with c_col2:
                    if st.button("🗑️", key=f"del_chat_{chat['id']}"):
                        if session_manager.delete_chat(st.session_state.session_id, chat['id']):
                            # Si borramos el activo, resetear
                            if is_active:
                                st.session_state.active_chat_id = None
                                st.session_state.chat_history = []
                            st.rerun()

        st.divider()
        st.subheader("Mis Proyectos")
        
        sessions = session_manager.list_sessions()
        for session in sessions:
            col_name, col_del = st.columns([4, 1])
            
            file_count = len(session.get('files', []))
            created_date = session.get('created_at', '')[:10]
            label = f"📂 {session['name']}\n{created_date}"
            
            with col_name:
                if st.button(label, key=f"sel_{session['id']}", use_container_width=True):
                    st.session_state.session_id = session['id']
                    st.session_state.chat_history = [] 
                    st.rerun()
            
            with col_del:
                if st.button("🗑️", key=f"del_{session['id']}"):
                    if session_manager.delete_session(session['id']):
                        if st.session_state.session_id == session['id']:
                            st.session_state.session_id = None
                            st.session_state.chat_history = []
                        st.rerun()
