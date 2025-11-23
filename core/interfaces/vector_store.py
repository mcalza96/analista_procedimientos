from abc import ABC, abstractmethod
from typing import List, Any, Tuple

class VectorStoreRepository(ABC):
    @abstractmethod
    def get_vector_db(self, session_path: str) -> Tuple[Any, Any]:
        pass

    @abstractmethod
    def add_documents(self, session_path: str, new_documents: List[Any]) -> Tuple[Any, Any]:
        pass

    @abstractmethod
    def clear_index(self, session_path: str) -> bool:
        """Elimina y limpia el índice vectorial y el almacenamiento de documentos."""
        pass
