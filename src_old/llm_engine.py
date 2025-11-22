import os
import logging
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_classic.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_classic.retrievers import EnsembleRetriever
from src.router import SemanticRouter

# Configuración de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Cargar variables de entorno
load_dotenv()

# Validación de API Key
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("❌ ERROR: No se encontró la variable de entorno GROQ_API_KEY.")

# ==================== TEMPLATES DE PROMPT POR RUTA ====================

PROMPT_PRECISION = """Eres un Asistente Técnico estricto de un laboratorio ISO 9001.
Tu tarea es responder la pregunta del usuario basándote ÚNICAMENTE en el contexto proporcionado.

Reglas:
1. Sé breve, directo y preciso (máximo 3 frases).
2. Responde SOLO con la información del contexto, sin elaborar.
3. Si no está la información exacta, di: "No figura en los procedimientos".
4. Cita el documento fuente usando referencias numéricas como [1], [2] al final de cada afirmación.
   NO escribas el nombre del archivo, solo el número.

Contexto:
{context}
"""

PROMPT_ANALYSIS = """Eres un Auditor de Calidad ISO 9001 estricto.
Tu objetivo es extraer evidencia concreta.
1. Si la respuesta NO está explícitamente en los documentos, di: 'No hay información suficiente en los procedimientos consultados'.
2. NO des recomendaciones generales si no están escritas en el texto.
3. Cita OBLIGATORIAMENTE el documento y la sección específica usando referencias numéricas como [1], [2].

Contexto:
{context}
"""

PROMPT_WALKTHROUGH = """Eres un Instructor de Laboratorio guiando un procedimiento paso a paso.
Tu objetivo es guiar al usuario a través del procedimiento descrito en el contexto, un paso a la vez.

Reglas:
1. Identifica el procedimiento relevante en el contexto.
2. Si el usuario dice "Empezar" o similar, presenta SOLO el Paso 1.
3. Si el usuario dice "Siguiente" o "Listo", presenta el siguiente paso.
4. Sé claro y conciso. Advierte sobre precauciones de seguridad si aparecen en el paso actual.
5. Usa referencias [1] si es necesario.

Contexto:
{context}
"""

PROMPT_CHAT = """Eres un asistente amable del laboratorio ISO 9001.
Responde al saludo o pregunta general de forma cordial y profesional.

Reglas:
1. Sé amable y profesional.
2. Si te preguntan sobre procedimientos específicos, sugiere al usuario que haga una pregunta más técnica o que suba los manuales pertinentes.
3. NO inventes información sobre procedimientos o documentos.
4. Mantén respuestas breves para saludos y agradecimientos.

Pregunta del usuario: {query}"""


def get_response(vectorstore: FAISS, bm25_retriever, query: str, chat_history: list, force_route: str = None) -> dict:
    """
    Genera una respuesta a una consulta utilizando Adaptive RAG con Enrutamiento Semántico y Búsqueda Híbrida.
    
    Args:
        vectorstore (FAISS): Base de datos vectorial.
        bm25_retriever: Retriever BM25 para búsqueda por palabras clave.
        query (str): Pregunta actual del usuario.
        chat_history (list): Historial de mensajes.
        force_route (str): Forzar una ruta específica (ej. WALKTHROUGH).

    Returns:
        dict: Respuesta completa con metadatos.
    """
    try:
        # ==================== PASO 1: CLASIFICACIÓN SEMÁNTICA ====================
        logger.info(f"📝 Procesando consulta: {query[:100]}...")
        
        if force_route:
            route = force_route
            logger.info(f"🚦 RUTA FORZADA: {route}")
        else:
            try:
                router = SemanticRouter()
                route = router.route(query)
                print(f"🚦 RUTA DETECTADA: {route}")
                logger.info(f"🚦 RUTA DETECTADA: {route}")
            except Exception as e:
                logger.error(f"⚠️ Error en clasificación semántica: {e}. Usando PRECISION por defecto.")
                route = "PRECISION"
        
        # ==================== PASO 2: EARLY RETURN - MODO CHAT ====================
        if route == "CHAT":
            print("💬 Modo CHAT activado (Sin búsqueda vectorial)")
            
            llm = ChatGroq(
                groq_api_key=GROQ_API_KEY,
                model_name="llama-3.3-70b-versatile",
                temperature=0.7
            )
            
            prompt_template = PromptTemplate.from_template(PROMPT_CHAT)
            chain = prompt_template | llm
            response_message = chain.invoke({"query": query})
            
            response_content = response_message.content if hasattr(response_message, 'content') else str(response_message)
            
            return {
                "result": response_content,
                "answer": response_content,
                "source_documents": [],
                "context": [],
                "route": route
            }
        
        # ==================== PASO 3: MODO RAG (PRECISION / ANALYSIS / WALKTHROUGH) ====================
        print(f"📚 Modo RAG activado: {route}")
        
        llm = ChatGroq(
            groq_api_key=GROQ_API_KEY,
            model_name="llama-3.3-70b-versatile",
            temperature=0.1  # Baja temperatura para fidelidad
        )
        
        # Configurar Retrievers Base
        faiss_retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 4})
        
        # Configurar Ensemble Retriever (Híbrido)
        # Pesos: 0.5 Vectorial (Semántico) + 0.5 BM25 (Palabras Clave)
        ensemble_retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, faiss_retriever],
            weights=[0.5, 0.5]
        )
        
        # Selección de Prompt y Configuración
        if route == "PRECISION":
            retriever = ensemble_retriever
            selected_prompt = PROMPT_PRECISION
            
        elif route == "ANALYSIS":
            # Para análisis, aumentamos k en FAISS (si fuera posible dinámicamente)
            # Como Ensemble es estático, usamos el mismo, pero el prompt pide más detalle.
            retriever = ensemble_retriever
            selected_prompt = PROMPT_ANALYSIS
            
        elif route == "WALKTHROUGH":
            retriever = ensemble_retriever
            selected_prompt = PROMPT_WALKTHROUGH
            
        else:
            retriever = ensemble_retriever
            selected_prompt = PROMPT_PRECISION

        # Sub-cadena 1: Contextualizar la pregunta
        contextualize_q_system_prompt = (
            "Given a chat history and the latest user question "
            "which might reference context in the chat history, "
            "formulate a standalone question which can be understood "
            "without the chat history. Do NOT answer the question, "
            "just reformulate it if needed and otherwise return it as is."
        )
        
        contextualize_q_prompt = ChatPromptTemplate.from_messages([
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])
        
        history_aware_retriever = create_history_aware_retriever(
            llm, retriever, contextualize_q_prompt
        )
        
        # Sub-cadena 2: Responder la pregunta
        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", selected_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])
        
        # Template para formatear documentos con índice para citas [1]
        # LangChain pasa 'context' como lista de docs. Necesitamos formatearlos con índices.
        # create_stuff_documents_chain usa document_prompt para cada doc.
        # No podemos inyectar el índice fácilmente en document_prompt estándar.
        # Solución: Usaremos un prompt simple y dejaremos que LangChain concatene, 
        # pero el LLM inferirá [1] basado en el orden o usaremos metadata si es posible.
        # Mejor aproximación para MVP: Incluir "Source: {source}" y pedir al LLM que cite.
        # Para que el LLM use [1], [2], idealmente deberíamos numerar los contextos en el prompt.
        # create_stuff_documents_chain no numera por defecto.
        # Sin embargo, podemos confiar en que el LLM vea "Source: X" y lo asocie.
        # Para MVP estricto de "NotebookLM style", necesitaríamos un custom chain, 
        # pero por ahora usaremos el nombre del archivo como referencia interna y el LLM generará [N].
        
        document_prompt = PromptTemplate(
            input_variables=["page_content", "source"],
            template="Content: {page_content}\nSource: {source}"
        )
        
        question_answer_chain = create_stuff_documents_chain(
            llm, 
            qa_prompt,
            document_prompt=document_prompt
        )
        
        rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
        
        response = rag_chain.invoke({
            "input": query,
            "chat_history": chat_history
        })
        
        # --- LOGS DE DEPURACIÓN ---
        source_docs = response.get("context", [])
        logger.info(f"📚 Documentos recuperados: {len(source_docs)}")
        
        # INYECCIÓN DE METADATOS
        response["route"] = route
        response["result"] = response["answer"]
        
        # FILTRADO DE FUENTES NEGATIVAS
        negative_phrases = [
            "no figura en los procedimientos",
            "no hay información suficiente",
            "no se encontró información"
        ]
        
        if any(phrase in response["answer"].lower() for phrase in negative_phrases):
            response["source_documents"] = []
        else:
            response["source_documents"] = response["context"]
        
        return response
    
    except Exception as e:
        logger.error(f"❌ Error generando respuesta con LLM: {e}")
        return {
            "result": "Lo siento, ocurrió un error al procesar tu solicitud.",
            "source_documents": [],
            "answer": "Error de procesamiento",
            "context": [],
            "route": "ERROR"
        }
