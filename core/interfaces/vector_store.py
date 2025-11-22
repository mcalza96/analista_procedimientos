from abc import ABC, abstractmethod
from typing import List, Any, Tuple

class VectorStoreRepository(ABC):
    @abstractmethod
    def get_vector_db(self, session_path: str) -> Tuple[Any, Any]:
        pass

    @abstractmethod
    def add_documents(self, session_path: str, new_documents: List[Any]) -> Tuple[Any, Any]:
        pass
