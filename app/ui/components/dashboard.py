import streamlit as st
import shutil
import os
from pathlib import Path
from langchain_core.messages import messages_to_dict
from typing import Any

def render_dashboard(
    session_manager: Any, 
    chat_service: Any, 
    session_name: str, 
    session_path: str, 
    is_draft: bool
):
    """
    Renderiza el panel principal con el título, controles de guardado y resumen de contexto.
    """
    # Header y Controles
    col_header, col_controls = st.columns([3, 1])

    with col_header:
        st.title(f"📑 {session_name}")

    # --- AUTOMATION: Generación Automática de Resumen ---
    if st.session_state.get("needs_summary_update"):
        if st.session_state.session_id and not is_draft:
            with st.spinner("🧠 Analizando nuevos documentos y actualizando contexto..."):
                try:
                    new_summary = chat_service.generate_context_summary()
                    session_manager.update_session_summary(st.session_state.session_id, new_summary)
                    # Limpiar flag
                    del st.session_state.needs_summary_update
                    st.rerun()
                except Exception as e:
                    st.error(f"Error actualizando resumen: {e}")
                    del st.session_state.needs_summary_update

    # --- Resumen de Contexto (Bajo Demanda) ---
    if st.session_state.session_id and not is_draft:
        # Intentar cargar resumen existente
        summary = session_manager.get_session_summary(st.session_state.session_id)
        
        if summary:
            with st.expander("🧠 Contexto del Proyecto (Resumen Ejecutivo)", expanded=True):
                st.markdown(summary)
                if st.button("🔄 Actualizar Análisis"):
                    with st.spinner("Re-analizando documentación..."):
                        new_summary = chat_service.generate_context_summary()
                        session_manager.update_session_summary(st.session_state.session_id, new_summary)
                        st.rerun()
        else:
            # Si hay archivos pero no resumen, ofrecer generarlo
            files_exist = len(session_manager.get_session_files(st.session_state.session_id)) > 0
            if files_exist:
                if st.button("🧠 Generar Resumen de Contexto", type="secondary"):
                    with st.spinner("Analizando base de conocimiento..."):
                        new_summary = chat_service.generate_context_summary()
                        session_manager.update_session_summary(st.session_state.session_id, new_summary)
                        st.rerun()

    with col_controls:
        if is_draft:
            with st.popover("💾 Guardar Proyecto", use_container_width=True):
                st.markdown("### Guardar Auditoría")
                new_name = st.text_input("Nombre del Proyecto")
                if st.button("Guardar", type="primary", use_container_width=True):
                    if new_name.strip():
                        new_id = session_manager.create_session(new_name)
                        new_path = session_manager.get_session_path(new_id)
                        
                        # Copiar datos del temporal al nuevo
                        try:
                            # Copiar raw_files si existen (CRÍTICO para evitar archivos fantasma)
                            if (Path(session_path) / "raw_files").exists():
                                shutil.copytree(
                                    Path(session_path) / "raw_files", 
                                    Path(new_path) / "raw_files", 
                                    dirs_exist_ok=True
                                )
                            
                            if (Path(session_path) / "vector_store").exists():
                                shutil.copytree(
                                    Path(session_path) / "vector_store", 
                                    Path(new_path) / "vector_store", 
                                    dirs_exist_ok=True
                                )
                            if (Path(session_path) / "doc_store").exists():
                                shutil.copytree(
                                    Path(session_path) / "doc_store", 
                                    Path(new_path) / "doc_store", 
                                    dirs_exist_ok=True
                                )
                                
                            # Si hay archivos en el borrador, registrarlos en la nueva sesión
                            # Esto es importante para que aparezcan en los metadatos
                            if (Path(session_path) / "raw_files").exists():
                                files = os.listdir(Path(session_path) / "raw_files")
                                valid_files = [f for f in files if not f.startswith('.')]
                                if valid_files:
                                    session_manager.add_files_to_session(new_id, valid_files)
                                    
                        except Exception as e:
                            st.error(f"Error migrando datos: {e}")

                        st.session_state.session_id = new_id
                        # Guardar historial actual
                        if st.session_state.chat_history:
                            history_dicts = messages_to_dict(st.session_state.chat_history)
                            session_manager.save_history(new_id, history_dicts)
                        
                        st.success("Proyecto guardado exitosamente.")
                        st.rerun()
                    else:
                        st.error("Nombre requerido.")
