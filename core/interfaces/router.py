from abc import ABC, abstractmethod

class RouterRepository(ABC):
    @abstractmethod
    def route_query(self, query: str) -> str:
        pass
