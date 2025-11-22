from typing import Generator
import logging
from langchain_groq import ChatGroq
from core.interfaces.llm_provider import LLMProvider
from config.settings import settings

logger = logging.getLogger(__name__)

class GroqProvider(LLMProvider):
    """
    Implementación del proveedor de LLM usando Groq.
    """
    def __init__(self) -> None:
        """Inicializa el cliente de Groq con la configuración definida."""
        if not settings.GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY not found")
        
        self.llm = ChatGroq(
            groq_api_key=settings.GROQ_API_KEY,
            model_name=settings.MODEL_NAME,
            temperature=0.1
        )

    def generate_response(self, prompt: str) -> str:
        """
        Genera una respuesta de texto basada en un prompt dado.
        
        Args:
            prompt: El texto del prompt completo.
            
        Returns:
            str: La respuesta generada por el modelo.
        """
        try:
            response = self.llm.invoke(prompt)
            return str(response.content)
        except Exception as e:
            logger.error(f"Error in generate_response: {e}")
            return "Error generando respuesta."

    def generate_stream(self, prompt: str) -> Generator[str, None, None]:
        """
        Genera un stream de respuesta de texto basada en un prompt dado.
        
        Args:
            prompt: El texto del prompt completo.
            
        Yields:
            str: Fragmentos de la respuesta generada.
        """
        try:
            for chunk in self.llm.stream(prompt):
                if chunk.content:
                    yield str(chunk.content)
        except Exception as e:
            logger.error(f"Error in generate_stream: {e}")
            yield "Error generando respuesta."
