import streamlit as st
from typing import Any
import tempfile

def render_home_view(session_manager: Any):
    """
    Renderiza la vista principal de selección de proyectos (Home).
    """
    st.title("🏠 Mis Proyectos")
    st.markdown("Gestiona tus auditorías y espacios de trabajo.")

    # --- CREAR NUEVO PROYECTO ---
    with st.container(border=True):
        col1, col2 = st.columns([3, 1])
        with col1:
            new_project_name = st.text_input("Nombre del nuevo proyecto", placeholder="Ej: Auditoría Interna 2024", label_visibility="collapsed")
        with col2:
            if st.button("➕ Crear Proyecto", type="primary", use_container_width=True):
                if new_project_name.strip():
                    session_id = session_manager.create_session(new_project_name)
                    st.session_state.session_id = session_id
                    st.session_state.chat_history = []
                    st.session_state.active_chat_id = None
                    st.rerun()
                else:
                    st.warning("El nombre no puede estar vacío.")

    st.divider()

    # --- LISTA DE PROYECTOS ---
    sessions = session_manager.list_sessions()
    
    if not sessions:
        st.info("No tienes proyectos creados aún. ¡Crea uno para empezar!")
    else:
        # Grid de tarjetas
        cols = st.columns(3)
        for i, session in enumerate(sessions):
            with cols[i % 3]:
                with st.container(border=True):
                    st.subheader(f"📂 {session['name']}")
                    
                    file_count = len(session.get('files', []))
                    created_date = session.get('created_at', '')[:10]
                    
                    st.caption(f"📅 {created_date} • 📄 {file_count} documentos")
                    
                    if session.get('summary'):
                        st.markdown(f"*{session['summary'][:100]}...*")
                    
                    col_open, col_del = st.columns([3, 1])
                    
                    with col_open:
                        if st.button("Abrir", key=f"open_{session['id']}", use_container_width=True):
                            st.session_state.session_id = session['id']
                            st.session_state.chat_history = []
                            st.session_state.active_chat_id = None
                            st.rerun()
                    
                    with col_del:
                        if st.button("🗑️", key=f"del_{session['id']}", help="Eliminar proyecto permanentemente"):
                            if session_manager.delete_session(session['id']):
                                st.rerun()
