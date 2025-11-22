from abc import ABC, abstractmethod
from typing import Generator

class LLMProvider(ABC):
    @abstractmethod
    def generate_response(self, prompt: str) -> str:
        pass

    @abstractmethod
    def generate_stream(self, prompt: str) -> Generator[str, None, None]:
        pass

