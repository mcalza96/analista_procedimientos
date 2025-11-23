from abc import ABC, abstractmethod

class FeedbackRepository(ABC):
    """Interfaz para el repositorio de feedback."""

    @abstractmethod
    def log_feedback(self, query: str, response: str, rating: str, details: str = ""):
        """Registra el feedback del usuario."""
        pass
