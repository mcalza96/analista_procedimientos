import streamlit as st
import os
from io import BytesIO
from gtts import gTTS
from langchain_core.messages import HumanMessage, AIMessage
from src.document_loader import process_documents
from src.vector_store import create_vector_db
from src.llm_engine import get_response
from src.utils import save_uploaded_files, log_feedback, render_page_image, display_sources

# Configuración de la página
st.set_page_config(
    page_title="Asistente de Laboratorio & Calidad",
    page_icon="🧪",
    layout="wide"
)

def get_chat_history_for_chain():
    """Convierte el historial de Streamlit al formato de LangChain."""
    history_langchain = []
    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            history_langchain.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            history_langchain.append(AIMessage(content=msg["content"]))
    return history_langchain

def generate_audio_briefing(text_content):
    """Genera un archivo de audio a partir de texto usando gTTS."""
    try:
        tts = gTTS(text=text_content, lang='es')
        fp = BytesIO()
        tts.write_to_fp(fp)
        fp.seek(0)
        return fp
    except Exception as e:
        st.error(f"Error generando audio: {e}")
        return None

def main():
    st.title("🧪 Asistente de Laboratorio & Calidad ISO 9001")
    st.markdown("""
    Este asistente utiliza Inteligencia Artificial para consultar tus procedimientos técnicos.
    Todas las respuestas están basadas estrictamente en los documentos proporcionados.
    """)

    # Inicialización del estado de la sesión
    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = None
    
    if "bm25_retriever" not in st.session_state:
        st.session_state.bm25_retriever = None
    
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    if "feedback_given" not in st.session_state:
        st.session_state.feedback_given = False
    
    if "show_feedback_form" not in st.session_state:
        st.session_state.show_feedback_form = False

    # --- BARRA LATERAL ---
    with st.sidebar:
        st.header("📂 Base de Conocimiento")
        st.info("Sube aquí tus procedimientos (PDF) para alimentar al asistente.")
        
        uploaded_files = st.file_uploader(
            "Cargar documentos PDF", 
            type="pdf", 
            accept_multiple_files=True
        )

        if st.button("Procesar Documentos", type="primary"):
            if uploaded_files:
                with st.spinner("⚙️ Procesando documentos... Esto puede tardar unos segundos."):
                    try:
                        # 1. Guardar archivos
                        pdf_paths = save_uploaded_files(uploaded_files)
                        
                        # 2. Cargar y dividir documentos
                        st.text("📄 Leyendo PDFs...")
                        chunks = process_documents(pdf_paths)
                        
                        # 3. Crear base de datos vectorial (Persistente + Híbrida)
                        st.text("🧠 Indexando conocimiento (Híbrido)...")
                        vectorstore, bm25_retriever = create_vector_db(chunks)
                        
                        # 4. Guardar en sesión
                        st.session_state.vectorstore = vectorstore
                        st.session_state.bm25_retriever = bm25_retriever
                        
                        # 5. Limpiar historial al cambiar documentos
                        st.session_state.chat_history = []
                        
                        st.success(f"✅ ¡Éxito! {len(chunks)} fragmentos indexados correctamente.")
                        
                    except Exception as e:
                        st.error(f"❌ Error durante el procesamiento: {str(e)}")
            else:
                # Intentar cargar índice existente si no se suben archivos
                if st.button("Cargar Índice Existente"):
                    try:
                        with st.spinner("🔄 Cargando índice guardado..."):
                            vectorstore, bm25_retriever = create_vector_db(chunks=[])
                            st.session_state.vectorstore = vectorstore
                            st.session_state.bm25_retriever = bm25_retriever
                            st.success("✅ Índice cargado desde disco.")
                    except Exception as e:
                        st.error(f"No se pudo cargar el índice: {e}")

        st.divider()
        
        # Audio Briefing
        if st.session_state.vectorstore:
            st.subheader("🎧 Audio Briefing")
            if st.button("Generar Resumen de Audio"):
                with st.spinner("Generando audio..."):
                    # Generar un resumen rápido (simulado o real)
                    # Para MVP, usamos un texto introductorio genérico + primer chunk
                    # Idealmente, pediríamos al LLM un resumen.
                    intro_text = "Bienvenido a su asistente de laboratorio. He procesado sus procedimientos. Estoy listo para ayudarle con consultas de seguridad y operación."
                    audio_fp = generate_audio_briefing(intro_text)
                    if audio_fp:
                        st.audio(audio_fp, format='audio/mp3')

    # --- ÁREA PRINCIPAL DE CHAT ---
    st.divider()

    if st.session_state.vectorstore is None:
        st.info("👈 **Para comenzar:** Por favor sube y procesa los procedimientos en el menú lateral.")
    else:
        # Botón de Modo Walkthrough
        col_mode1, col_mode2 = st.columns([1, 5])
        with col_mode1:
            if st.button("🚀 Modo Guía"):
                st.session_state.chat_history.append({
                    "role": "user", 
                    "content": "Guíame paso a paso por el procedimiento principal.",
                    "force_route": "WALKTHROUGH"
                })
                st.rerun()

        # 1. Mostrar historial de mensajes
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                # Mostrar indicador visual de estrategia para mensajes del asistente
                if message["role"] == "assistant":
                    route = message.get("route", None)
                    
                    # Mostrar badge de estrategia según la ruta
                    if route == "PRECISION":
                        st.caption("🔎 **Estrategia:** Búsqueda de datos específicos")
                    elif route == "ANALYSIS":
                        st.caption("🧠 **Estrategia:** Análisis profundo de documentos")
                    elif route == "WALKTHROUGH":
                        st.caption("👣 **Estrategia:** Guía Paso a Paso")
                    elif route == "CHAT":
                        st.caption("💬 **Estrategia:** Modo conversación")
                
                # Mostrar contenido del mensaje
                st.markdown(message["content"])
                
                # Si hay fuentes guardadas Y la ruta NO es CHAT, mostrarlas
                if "sources" in message and message.get("route") != "CHAT":
                    display_sources(message["sources"], message["content"])

        # 2. Input del usuario
        if prompt := st.chat_input("Escribe tu pregunta sobre los procedimientos..."):
            # Mostrar mensaje del usuario inmediatamente
            st.chat_message("user").markdown(prompt)
            # Guardar en historial
            st.session_state.chat_history.append({"role": "user", "content": prompt})

            # 3. Generar respuesta
            with st.chat_message("assistant"):
                with st.spinner("Analizando procedimientos..."):
                    # Obtener historial formateado para LangChain
                    history_langchain = get_chat_history_for_chain()
                    
                    # Verificar si el último mensaje del usuario forzó una ruta
                    last_msg = st.session_state.chat_history[-1]
                    force_route = last_msg.get("force_route", None)

                    # Llamada al motor de IA
                    response_dict = get_response(
                        st.session_state.vectorstore, 
                        st.session_state.bm25_retriever,
                        prompt, 
                        history_langchain,
                        force_route=force_route
                    )
                    
                    answer = response_dict.get("result", "Lo siento, no pude generar una respuesta.")
                    source_docs = response_dict.get("source_documents", [])
                    route = response_dict.get("route", None)
                    
                    # Mostrar indicador visual de estrategia
                    if route == "PRECISION":
                        st.caption("🔎 **Estrategia:** Búsqueda de datos específicos")
                    elif route == "ANALYSIS":
                        st.caption("🧠 **Estrategia:** Análisis profundo de documentos")
                    elif route == "WALKTHROUGH":
                        st.caption("👣 **Estrategia:** Guía Paso a Paso")
                    elif route == "CHAT":
                        st.caption("💬 **Estrategia:** Modo conversación")
                    
                    st.markdown(answer)
                    
                    # Mostrar fuentes solo si NO es CHAT y hay documentos
                    if route != "CHAT" and source_docs:
                        display_sources(source_docs, answer)
            
            # Guardar respuesta en historial
            st.session_state.chat_history.append({
                "role": "assistant", 
                "content": answer,
                "sources": source_docs,
                "route": route
            })
            
            # Resetear estado de feedback para la nueva respuesta
            st.session_state.feedback_given = False
            st.session_state.show_feedback_form = False
            
            # Forzar rerun para mostrar el widget de feedback
            st.rerun()
        
        # 4. Widget de Feedback (Solo para la última respuesta del asistente)
        if len(st.session_state.chat_history) > 0 and st.session_state.chat_history[-1]["role"] == "assistant":
            if not st.session_state.feedback_given:
                st.divider()
                st.caption("¿Te sirvió esta conversación?")
                
                # Widget de feedback con thumbs
                feedback_value = st.feedback("thumbs", key="feedback_widget")
                
                # Procesar feedback positivo (👍)
                if feedback_value == 1:  # Thumbs up
                    # Obtener la última interacción
                    last_user_msg = None
                    last_assistant_msg = st.session_state.chat_history[-1]
                    
                    # Buscar la última pregunta del usuario
                    for i in range(len(st.session_state.chat_history) - 1, -1, -1):
                        if st.session_state.chat_history[i]["role"] == "user":
                            last_user_msg = st.session_state.chat_history[i]
                            break
                    
                    if last_user_msg:
                        log_feedback(
                            query=last_user_msg["content"],
                            response=last_assistant_msg["content"],
                            rating="Positiva",
                            details=""
                        )
                        st.toast("✅ ¡Gracias por tu feedback! Ayudas a mejorar el sistema.", icon="👍")
                        st.session_state.feedback_given = True
                        st.rerun()
                
                # Procesar feedback negativo (👎)
                elif feedback_value == 0:  # Thumbs down
                    st.session_state.show_feedback_form = True
                
                # Mostrar formulario de detalles para feedback negativo
                if st.session_state.show_feedback_form:
                    st.warning("⚠️ Lamentamos que la respuesta no haya sido útil.")
                    
                    error_details = st.text_input(
                        "¿Qué falló? (Ej: Procedimiento incorrecto, alucinación, información incompleta...)",
                        key="error_details_input",
                        placeholder="Describe brevemente el problema..."
                    )
                    
                    col1, col2 = st.columns([1, 4])
                    with col1:
                        if st.button("📤 Enviar Reporte", type="primary"):
                            # Obtener la última interacción
                            last_user_msg = None
                            last_assistant_msg = st.session_state.chat_history[-1]
                            
                            # Buscar la última pregunta del usuario
                            for i in range(len(st.session_state.chat_history) - 1, -1, -1):
                                if st.session_state.chat_history[i]["role"] == "user":
                                    last_user_msg = st.session_state.chat_history[i]
                                    break
                            
                            if last_user_msg:
                                log_feedback(
                                    query=last_user_msg["content"],
                                    response=last_assistant_msg["content"],
                                    rating="Negativa",
                                    details=error_details
                                )
                                st.success("✅ Reporte enviado. Trabajaremos en mejorar el sistema.")
                                st.session_state.feedback_given = True
                                st.session_state.show_feedback_form = False
                                st.rerun()
                    
                    with col2:
                        if st.button("Cancelar"):
                            st.session_state.show_feedback_form = False
                            st.rerun()

if __name__ == "__main__":
    main()
