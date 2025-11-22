import logging
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from config.settings import settings

logger = logging.getLogger(__name__)

class SemanticRouter:
    def __init__(self):
        try:
            if not settings.GROQ_API_KEY:
                raise ValueError("GROQ_API_KEY not found")

            self.llm = ChatGroq(
                groq_api_key=settings.GROQ_API_KEY,
                model_name=settings.MODEL_NAME,
                temperature=0
            )
            
            self.classification_prompt = PromptTemplate(
                input_variables=["query"],
                template="""Eres un clasificador de preguntas experto. Tu tarea es analizar la siguiente pregunta y clasificarla en una de estas tres categorías ÚNICAMENTE:

PRECISION: Si la pregunta solicita:
- Un dato específico, valor concreto o número exacto
- El nombre de un responsable, cargo o persona
- Una referencia puntual a un documento, sección o procedimiento
- Información precisa y específica
- Preguntas de seguimiento cortas que buscan un dato concreto (ej: "¿y cuál es el plazo?", "¿quién lo firma?")

ANALYSIS: Si la pregunta solicita:
- Un resumen de información
- Una comparación entre elementos
- Una explicación de un proceso o procedimiento
- Una auditoría o análisis de cumplimiento
- Interpretación o síntesis de información
- Preguntas de seguimiento que requieren contexto o explicación (ej: "¿por qué?", "explícame más", "¿qué significa eso?")

CHAT: Si la pregunta es:
- Un saludo (hola, buenos días, etc.)
- Un agradecimiento (gracias, muchas gracias, etc.)
- Una pregunta TOTALMENTE fuera del contexto documental o de procedimientos (ej: "¿qué hora es?", "cuéntame un chiste")
- Conversación informal SIN relación con información técnica

IMPORTANTE:
- Si la pregunta menciona términos técnicos (como "lodo", "ISO", "procedimiento", "auditoría", etc.), NUNCA la clasifiques como CHAT.
- Si la pregunta parece una continuación de una conversación técnica (empieza con "pero", "y", "entonces"), clasifícala como ANALYSIS o PRECISION, NO como CHAT.
- Responde ÚNICAMENTE con una de estas tres palabras: PRECISION, ANALYSIS, o CHAT. No incluyas explicaciones, puntuación adicional ni ningún otro texto.

Pregunta: {query}

Clasificación:"""
            )
            
            logger.info("✅ SemanticRouter inicializado correctamente")
            
        except Exception as e:
            logger.error(f"❌ Error inicializando SemanticRouter: {e}")
            raise
    
    def route(self, query: str) -> str:
        try:
            if not query or not isinstance(query, str):
                return "PRECISION"
            
            formatted_prompt = self.classification_prompt.format(query=query)
            response = self.llm.invoke(formatted_prompt)
            classification = response.content.strip().upper()
            
            valid_categories = ["PRECISION", "ANALYSIS", "CHAT"]
            if classification not in valid_categories:
                return "PRECISION"
            
            return classification
            
        except Exception as e:
            logger.error(f"❌ Error en clasificación semántica: {e}")
            return "PRECISION"
