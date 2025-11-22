from abc import ABC, abstractmethod
from typing import List, Dict, Any
from core.domain.models import ChatResponse, Quiz

class LLMProvider(ABC):
    @abstractmethod
    def generate_response(self, query: str, context: List[Any], chat_history: List[Any], route: str) -> ChatResponse:
        pass

    @abstractmethod
    def generate_quiz(self, topic: str, difficulty: str, num_questions: int, context: List[Any]) -> Quiz:
        pass
