from core.interfaces.llm_provider import LLMProvider
from core.domain.models import Quiz
from typing import Any

class QuizService:
    def __init__(self, llm_provider: LLMProvider, chat_service: Any):
        self.llm_provider = llm_provider
        self.chat_service = chat_service # To access vector store

    def generate_quiz(self, topic: str, difficulty: str, num_questions: int) -> Quiz:
        context = (self.chat_service.vector_store, self.chat_service.bm25_retriever)
        return self.llm_provider.generate_quiz(topic, difficulty, num_questions, context)
