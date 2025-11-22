import streamlit as st
from core.services.chat_service import ChatService
from app.ui.components.pdf_viewer import display_sources
from infrastructure.logging.feedback_logger import FeedbackLogger
from langchain_core.messages import HumanMessage, AIMessage

def render_chat_view(chat_service: ChatService):
    st.header("💬 Asistente ISO 9001")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    for i, message in enumerate(st.session_state.chat_history):
        if isinstance(message, HumanMessage):
            with st.chat_message("user"):
                st.markdown(message.content)
        elif isinstance(message, AIMessage):
            with st.chat_message("assistant"):
                st.markdown(message.content)
                if hasattr(message, "additional_kwargs") and "sources" in message.additional_kwargs:
                     display_sources(message.additional_kwargs["sources"], message.content)
                
                if i == len(st.session_state.chat_history) - 1:
                    col1, col2 = st.columns([1, 1])
                    with col1:
                        if st.button("👍", key=f"up_{i}"):
                            FeedbackLogger.log_feedback(
                                st.session_state.chat_history[i-1].content, 
                                message.content, 
                                "Positiva"
                            )
                            st.toast("¡Gracias por tu feedback!")
                    with col2:
                        if st.button("👎", key=f"down_{i}"):
                            FeedbackLogger.log_feedback(
                                st.session_state.chat_history[i-1].content, 
                                message.content, 
                                "Negativa"
                            )
                            st.toast("Feedback registrado.")

    if prompt := st.chat_input("¿En qué puedo ayudarte hoy?"):
        st.session_state.chat_history.append(HumanMessage(content=prompt))
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Analizando documentos..."):
                history_for_chain = st.session_state.chat_history[:-1]
                response = chat_service.get_response(prompt, history_for_chain)
                
                st.markdown(response.answer)
                display_sources(response.source_documents, response.answer)
                
                ai_msg = AIMessage(content=response.answer)
                ai_msg.additional_kwargs["sources"] = response.source_documents
                st.session_state.chat_history.append(ai_msg)
                st.rerun()
