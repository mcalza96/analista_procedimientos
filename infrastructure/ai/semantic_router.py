import logging
from functools import lru_cache
from langchain_groq import ChatGroq
from config.settings import settings
from core.interfaces.router import RouterRepository
from core.services.prompt_manager import PromptManager
from infrastructure.constants import ROUTER_CATEGORIES

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
            
            self.prompt_manager = PromptManager()
            logger.info("✅ SemanticRouter inicializado correctamente")
            
        except Exception as e:
            logger.error(f"❌ Error inicializando SemanticRouter: {e}")
            raise
    
    @lru_cache(maxsize=100)
    def route_query(self, query: str) -> str:
        try:
            if not query or not isinstance(query, str):
                return ROUTER_CATEGORIES[0]
            
            prompt = self.prompt_manager.get_classification_prompt(query)
            response = self.llm.invoke(prompt)
            classification = response.content.strip().upper()
            
            if classification not in ROUTER_CATEGORIES:
                return ROUTER_CATEGORIES[0]
            
            return classification
            
        except Exception as e:
            logger.error(f"❌ Error en clasificación semántica: {e}")
            return ROUTER_CATEGORIES[0]
