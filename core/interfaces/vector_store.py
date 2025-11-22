from abc import ABC, abstractmethod
from typing import List, Any, Tuple

class VectorStoreRepository(ABC):
    @abstractmethod
    def create_vector_db(self, chunks: List[Any]) -> Tuple[Any, Any]:
        pass

    @abstractmethod
    def get_retriever(self, vectorstore: Any, bm25_retriever: Any):
        pass
