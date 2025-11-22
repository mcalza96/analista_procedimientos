from abc import ABC, abstractmethod
from typing import List, Any

class DocumentLoaderRepository(ABC):
    @abstractmethod
    def load_documents(self, file_paths: List[str]) -> List[Any]:
        pass
