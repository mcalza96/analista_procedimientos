from typing import List, Any
from core.interfaces.llm_provider import LLMProvider
from core.interfaces.vector_store import VectorStoreRepository
from infrastructure.files.loader import DocumentLoader
from core.domain.models import ChatResponse

class ChatService:
    def __init__(self, llm_provider: LLMProvider, vector_store_repo: VectorStoreRepository):
        self.llm_provider = llm_provider
        self.vector_store_repo = vector_store_repo
        self.vector_store = None
        self.bm25_retriever = None

    def process_uploaded_files(self, file_paths: List[str]):
        chunks = DocumentLoader.process_documents(file_paths)
        self.vector_store, self.bm25_retriever = self.vector_store_repo.create_vector_db(chunks)

    def load_existing_db(self):
        self.vector_store, self.bm25_retriever = self.vector_store_repo.create_vector_db([])

    def get_response(self, query: str, chat_history: List[Any], route: str = None) -> ChatResponse:
        context = (self.vector_store, self.bm25_retriever)
        return self.llm_provider.generate_response(query, context, chat_history, route)
