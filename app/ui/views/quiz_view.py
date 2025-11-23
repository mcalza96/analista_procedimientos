import streamlit as st
import json
from core.services.chat_service import ChatService

def render_quiz_view(chat_service: ChatService):
    """
    Renderiza la vista del generador de cuestionarios.
    """
    st.title("🎓 Generador de Evaluaciones ISO 9001")
    st.markdown("Genera exámenes automáticos basados en tu documentación para entrenamiento y auditoría.")

    # Inicializar estado del quiz
    if "quiz_data" not in st.session_state:
        st.session_state.quiz_data = None
    if "quiz_answers" not in st.session_state:
        st.session_state.quiz_answers = {}
    if "quiz_submitted" not in st.session_state:
        st.session_state.quiz_submitted = False

    # --- VISTA 1: CONFIGURACIÓN ---
    if not st.session_state.quiz_data:
        with st.container(border=True):
            st.subheader("⚙️ Configuración del Examen")
            
            col1, col2 = st.columns(2)
            with col1:
                topic = st.text_input("Tema a evaluar", placeholder="Ej: Auditoría Interna, Control de Documentos...")
                difficulty = st.selectbox("Nivel de Dificultad", ["Básico", "Intermedio", "Avanzado"])
            
            with col2:
                num_questions = st.slider("Cantidad de Preguntas", min_value=3, max_value=10, value=5)
            
            if st.button("🚀 Generar Examen", type="primary", use_container_width=True):
                if not topic:
                    st.warning("Por favor ingresa un tema.")
                    return

                with st.spinner("Analizando documentación y generando preguntas..."):
                    try:
                        json_response = chat_service.generate_quiz(topic, difficulty, num_questions)
                        data = json.loads(json_response)
                        
                        if "questions" in data:
                            st.session_state.quiz_data = data
                            st.session_state.quiz_answers = {}
                            st.session_state.quiz_submitted = False
                            st.rerun()
                        else:
                            st.error("El formato del cuestionario generado no es válido.")
                            st.json(data) # Debug
                            
                    except json.JSONDecodeError:
                        st.error("Error al procesar la respuesta del modelo. Intenta de nuevo.")
                    except Exception as e:
                        st.error(f"Error inesperado: {e}")

    # --- VISTA 2: TOMAR EL EXAMEN ---
    else:
        data = st.session_state.quiz_data
        questions = data.get("questions", [])
        
        # Header con botón de volver
        col_h1, col_h2 = st.columns([4, 1])
        with col_h1:
            st.subheader(f"📝 Examen: {data.get('topic', 'General')}")
        with col_h2:
            if st.button("🔄 Nuevo", use_container_width=True):
                st.session_state.quiz_data = None
                st.session_state.quiz_answers = {}
                st.session_state.quiz_submitted = False
                st.rerun()

        # Renderizar preguntas
        score = 0
        
        for idx, q in enumerate(questions):
            with st.container(border=True):
                st.markdown(f"**{idx + 1}. {q['question']}**")
                
                options = q['options']
                # Clave única para el widget
                key = f"q_{idx}"
                
                # Recuperar respuesta previa si existe
                user_idx = st.session_state.quiz_answers.get(idx, None)
                
                # Si ya se envió, mostramos feedback visual
                if st.session_state.quiz_submitted:
                    # Mostrar opción seleccionada (deshabilitada)
                    st.radio(
                        "Tu respuesta:", 
                        options, 
                        index=user_idx if user_idx is not None else 0,
                        key=f"disabled_{idx}",
                        disabled=True,
                        label_visibility="collapsed"
                    )
                    
                    correct_idx = q['correct_answer']
                    is_correct = (user_idx == correct_idx)
                    
                    if is_correct:
                        st.success("✅ ¡Correcto!")
                        score += 1
                    else:
                        st.error(f"❌ Incorrecto. La respuesta correcta era: **{options[correct_idx]}**")
                    
                    st.info(f"💡 **Explicación:** {q['explanation']}")
                    if q.get('source_file'):
                        st.caption(f"📖 Fuente: {q['source_file']} (Pág. {q.get('page_number', 'N/A')})")
                        
                else:
                    # Modo selección
                    selected_opt = st.radio(
                        "Selecciona una opción:", 
                        options, 
                        index=None, 
                        key=key,
                        label_visibility="collapsed"
                    )
                    
                    # Guardar selección en estado
                    if selected_opt:
                        # Encontrar índice
                        try:
                            st.session_state.quiz_answers[idx] = options.index(selected_opt)
                        except ValueError:
                            pass

        # Botón de envío
        if not st.session_state.quiz_submitted:
            if st.button("✅ Finalizar y Calificar", type="primary", use_container_width=True):
                # Verificar que todas las preguntas tengan respuesta (opcional, o contar como malas)
                if len(st.session_state.quiz_answers) < len(questions):
                    st.warning("⚠️ Aún tienes preguntas sin responder.")
                else:
                    st.session_state.quiz_submitted = True
                    st.rerun()
        else:
            # Mostrar resultado final
            final_score = (score / len(questions)) * 100
            if final_score >= 80:
                st.balloons()
                st.success(f"🎉 **Resultado Final: {score}/{len(questions)} ({final_score:.0f}%)** - ¡Aprobado!")
            else:
                st.warning(f"📊 **Resultado Final: {score}/{len(questions)} ({final_score:.0f}%)** - Necesitas repasar.")
