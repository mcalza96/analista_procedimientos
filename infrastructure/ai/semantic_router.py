import logging
from langchain_groq import ChatGroq
from config.settings import settings
from core.interfaces.router import RouterRepository
from core.services.prompt_manager import PromptManager

logger = logging.getLogger(__name__)

class SemanticRouter(RouterRepository):
    def __init__(self):
        try:
            if not settings.GROQ_API_KEY:
                raise ValueError("GROQ_API_KEY not found")

            self.llm = ChatGroq(
                groq_api_key=settings.GROQ_API_KEY,
                model_name=settings.MODEL_NAME,
                temperature=0
            )
            
            logger.info("✅ SemanticRouter inicializado correctamente")
            
        except Exception as e:
            logger.error(f"❌ Error inicializando SemanticRouter: {e}")
            raise
    
    def route_query(self, query: str) -> str:
        try:
            if not query or not isinstance(query, str):
                return "PRECISION"
            
            prompt = PromptManager.get_classification_prompt(query)
            response = self.llm.invoke(prompt)
            classification = response.content.strip().upper()
            
            valid_categories = ["PRECISION", "ANALYSIS", "CHAT"]
            if classification not in valid_categories:
                return "PRECISION"
            
            return classification
            
        except Exception as e:
            logger.error(f"❌ Error en clasificación semántica: {e}")
            return "PRECISION"
