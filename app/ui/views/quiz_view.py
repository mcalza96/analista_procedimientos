import streamlit as st
import os
from streamlit_pdf_viewer import pdf_viewer
from config.settings import settings
from core.services.quiz_service import QuizService

def render_quiz_view(quiz_service: QuizService):
    st.header("🎓 Entrenador de Personal (Quiz)")

    if "quiz_data" not in st.session_state:
        st.session_state.quiz_data = None
    if "quiz_answers" not in st.session_state:
        st.session_state.quiz_answers = {}
    if "quiz_submitted" not in st.session_state:
        st.session_state.quiz_submitted = False

    with st.sidebar:
        st.subheader("Configuración del Examen")
        topic = st.text_input("Tema del examen", "Procedimientos de Seguridad")
        difficulty = st.selectbox("Dificultad", ["Fácil", "Medio", "Difícil"])
        num_questions = st.slider("Número de preguntas", 3, 10, 5)
        
        if st.button("Generar Examen"):
            with st.spinner("Generando preguntas..."):
                quiz = quiz_service.generate_quiz(topic, difficulty, num_questions)
                st.session_state.quiz_data = quiz
                st.session_state.quiz_answers = {}
                st.session_state.quiz_submitted = False
                st.rerun()

    if st.session_state.quiz_data:
        quiz = st.session_state.quiz_data
        st.subheader(f"Examen: {quiz.topic}")
        
        with st.form("quiz_form"):
            for i, q in enumerate(quiz.questions):
                st.markdown(f"**{i+1}. {q.question}**")
                
                # Restore previous answer if exists
                key = f"q_{i}"
                
                # Radio button for options
                # We need to map the selected string back to index or use index as value
                # Streamlit radio returns the label by default.
                selected_option = st.radio(
                    "Selecciona una opción:",
                    q.options,
                    key=key,
                    index=None,
                    label_visibility="collapsed"
                )
                
            submitted = st.form_submit_button("Enviar Respuestas")
            
            if submitted:
                st.session_state.quiz_submitted = True
                # Rerun to show results
                st.rerun()

        if st.session_state.quiz_submitted:
            st.divider()
            st.subheader("Resultados")
            score = 0
            for i, q in enumerate(quiz.questions):
                # Get user answer from session state
                user_answer_label = st.session_state.get(f"q_{i}")
                
                if user_answer_label:
                    try:
                        user_idx = q.options.index(user_answer_label)
                    except ValueError:
                        user_idx = -1
                else:
                    user_idx = -1

                is_correct = (user_idx == q.correct_answer)
                if is_correct:
                    score += 1
                    st.success(f"✅ Pregunta {i+1}: Correcta")
                else:
                    correct_label = q.options[q.correct_answer]
                    st.error(f"❌ Pregunta {i+1}: Incorrecta. La respuesta correcta era: {correct_label}")
                    st.info(f"💡 Explicación: {q.explanation}")
                
                # Safely access source_file and page_number to handle stale session state
                source_file = getattr(q, 'source_file', None)
                page_number = getattr(q, 'page_number', None)
                
                # DEBUG: Ver qué está llegando realmente
                # st.caption(f"🕵️ Debug: Archivo='{source_file}', Pág='{page_number}'")

                if source_file:
                    with st.popover("🔍 Ver Fuente Original"):
                        full_path = os.path.join(settings.TEMP_DOCS_DIR, source_file)
                        if os.path.exists(full_path):
                            # page_number comes 0-indexed from metadata usually, check consistency
                            # In GroqProvider we inject 'Page: {page}' where page is from metadata (0-indexed)
                            # But LLM might return it as is.
                            # pdf_viewer expects 1-indexed list for pages_to_render? No, let's check pdf_viewer docs or usage.
                            # In pdf_viewer.py we used: pages_to_render=[page], where page was doc.metadata.get('page', 0) + 1
                            # So pdf_viewer likely expects 1-based index if we want to match visual page number.
                            # Let's assume LLM returns the number it saw in context.
                            # In context we injected: Page: {d.metadata.get('page', 0)} (which is 0-indexed)
                            # So LLM returns 0-indexed page.
                            # To render, we might need to pass [page + 1] if pdf_viewer expects 1-based.
                            # Let's try passing [q.page_number + 1] to be safe and consistent with other viewer.
                            
                            page_to_show = (page_number if page_number is not None else 0) + 1
                            st.caption(f"Fuente: {source_file} (Pág. {page_to_show})")
                            
                            pdf_viewer(
                                full_path, 
                                width=600, 
                                height=500, 
                                pages_to_render=[page_to_show]
                            )
                        else:
                            st.warning(f"Archivo fuente no encontrado: {source_file}")

            final_score = (score / len(quiz.questions)) * 100
            st.metric("Calificación Final", f"{final_score:.1f}%")
            
    else:
        st.info("Configura y genera un examen desde la barra lateral para comenzar.")
