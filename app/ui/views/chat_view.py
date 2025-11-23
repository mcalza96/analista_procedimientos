import streamlit as st
import os
from core.services.chat_service import ChatService
from core.services.document_service import DocumentService
from core.interfaces.vector_store import VectorStoreRepository
from core.interfaces.feedback_repository import FeedbackRepository
from langchain_core.messages import HumanMessage, AIMessage

def _render_sources(source_documents):
    """Renderiza las fuentes documentales con estilo profesional."""
    if not source_documents:
        return
    
    st.markdown("---")
    st.markdown("#### 📚 Referencias Documentales")
    
    for i, doc in enumerate(source_documents, 1):
        # Manejo flexible de objetos SourceDocument o diccionarios (desde historial)
        content = ""
        metadata = {}
        
        if hasattr(doc, 'page_content'):
            content = doc.page_content
            metadata = doc.metadata
        elif isinstance(doc, dict):
            content = doc.get('page_content', '')
            metadata = doc.get('metadata', {})
        
        # Limpieza del nombre del archivo
        source_path = metadata.get('source_file', 'Documento Desconocido')
        filename = os.path.basename(source_path)
            
        page = metadata.get('page', 'N/A')
        # Algunos metadatos pueden tener 'page_number' en lugar de 'page'
        if page == 'N/A':
            page = metadata.get('page_number', 'N/A')
            
        score = metadata.get('score', None)
        
        label = f"📄 [{i}] {filename} (Pág. {page})"
        
        with st.expander(label):
            st.markdown(f"> {content}")
            if score:
                st.caption(f"🎯 Relevancia: {score:.4f}")

def render_chat_view(
    chat_service: ChatService, 
    doc_service: DocumentService, 
    vector_repo: VectorStoreRepository, 
    feedback_logger: FeedbackRepository,
    session_path: str
):
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # --- HEADER DE ACCIONES DEL CHAT ---
    if st.session_state.chat_history:
        col_h1, col_h2 = st.columns([4, 1])
        with col_h2:
            if st.button("🧠 Guardar Conversación", help="Guardar todo el chat actual como un documento de conocimiento"):
                # Formatear conversación completa
                full_text = "REGISTRO DE CONVERSACIÓN\n========================\n\n"
                for msg in st.session_state.chat_history:
                    role = "USUARIO" if isinstance(msg, HumanMessage) else "ASISTENTE"
                    full_text += f"[{role}]: {msg.content}\n\n"
                
                # Título automático
                title = f"Chat Guardado {len(st.session_state.chat_history)//2} interacciones"
                
                with st.spinner("Indexando conversación..."):
                    success = doc_service.ingest_text_as_document(
                        text_content=full_text,
                        title=title,
                        session_path=session_path,
                        vector_repo=vector_repo
                    )
                    
                    if success:
                        st.toast("✅ Conversación guardada en memoria", icon="🧠")
                        # Opcional: Trigger update summary
                        st.session_state.needs_summary_update = True
                    else:
                        st.error("Error al guardar conversación.")

    for i, message in enumerate(st.session_state.chat_history):
        if isinstance(message, HumanMessage):
            with st.chat_message("user", avatar="👤"):
                st.markdown(message.content)
        elif isinstance(message, AIMessage):
            with st.chat_message("assistant", avatar="🤖"):
                st.markdown(message.content)
                
                # Renderizar fuentes si existen
                if hasattr(message, "additional_kwargs") and "sources" in message.additional_kwargs:
                     _render_sources(message.additional_kwargs["sources"])
                
                # Botones de acción (Feedback y Guardar)
                if i == len(st.session_state.chat_history) - 1:
                    col1, col2, col3 = st.columns([1, 1, 2])
                    with col1:
                        if st.button("👍", key=f"up_{i}"):
                            feedback_logger.log_feedback(
                                st.session_state.chat_history[i-1].content, 
                                message.content, 
                                "Positiva"
                            )
                            st.toast("¡Gracias por tu feedback!")
                    with col2:
                        if st.button("👎", key=f"down_{i}"):
                            feedback_logger.log_feedback(
                                st.session_state.chat_history[i-1].content, 
                                message.content, 
                                "Negativa"
                            )
                            st.toast("Feedback registrado.")
                    
                    # Botón de Aprendizaje Activo
                    with col3:
                        if st.button("🧠 Guardar como Conocimiento", key=f"save_{i}"):
                            last_question = st.session_state.chat_history[i-1].content
                            answer_to_save = message.content
                            
                            formatted_content = f"PREGUNTA: {last_question}\n\nRESPUESTA VALIDADA: {answer_to_save}"
                            # Título corto: primeros 30 chars de la pregunta
                            title = (last_question[:30] + '..') if len(last_question) > 30 else last_question
                            
                            success = doc_service.ingest_text_as_document(
                                text_content=formatted_content,
                                title=title,
                                session_path=session_path,
                                vector_repo=vector_repo
                            )
                            
                            if success:
                                st.success("✅ Conocimiento guardado exitosamente en la base de datos.")
                            else:
                                st.error("❌ Error al guardar el conocimiento.")

    if prompt := st.chat_input("¿En qué puedo ayudarte hoy?"):
        st.session_state.chat_history.append(HumanMessage(content=prompt))
        with st.chat_message("user", avatar="👤"):
            st.markdown(prompt)

        with st.chat_message("assistant", avatar="🤖"):
            # Placeholder for status
            status_placeholder = st.empty()
            full_response = ""
            source_docs = []
            
            with status_placeholder.status("Consultando base de conocimiento...", expanded=True) as status:
                history_for_chain = st.session_state.chat_history[:-1]
                # Use streaming response
                response_generator, source_docs, route = chat_service.get_streaming_response(prompt, history_for_chain)
                status.update(label="Generando respuesta...", state="running")
                
                # Stream the response
                full_response = st.write_stream(response_generator)
                status.update(label="¡Respuesta completada!", state="complete", expanded=False)
            
            # Display sources after generation
            _render_sources(source_docs)
            
            # Save to history
            ai_msg = AIMessage(content=full_response)
            ai_msg.additional_kwargs["sources"] = source_docs
            st.session_state.chat_history.append(ai_msg)
            
            # Force rerun to show feedback buttons and update history view
            st.rerun()
