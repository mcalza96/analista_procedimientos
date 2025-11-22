from typing import List, Any
from core.interfaces.llm_provider import LLMProvider
from core.interfaces.vector_store import VectorStoreRepository
from core.interfaces.document_loader import DocumentLoaderRepository
from core.interfaces.router import RouterRepository
from core.domain.models import ChatResponse

class ChatService:
    def __init__(
        self, 
        llm_provider: LLMProvider, 
        vector_store_repo: VectorStoreRepository,
        document_loader: DocumentLoaderRepository,
        router_repo: RouterRepository
    ):
        self.llm_provider = llm_provider
        self.vector_store_repo = vector_store_repo
        self.document_loader = document_loader
        self.router_repo = router_repo
        self.vector_store = None
        self.bm25_retriever = None

    def process_uploaded_files(self, file_paths: List[str]):
        chunks = self.document_loader.load_documents(file_paths)
        self.vector_store, self.bm25_retriever = self.vector_store_repo.create_vector_db(chunks)

    def load_existing_db(self):
        self.vector_store, self.bm25_retriever = self.vector_store_repo.create_vector_db([])

    def get_response(self, query: str, chat_history: List[Any], route: str = None) -> ChatResponse:
        if route is None:
            route = self.router_repo.route_query(query)
            
        context = (self.vector_store, self.bm25_retriever)
        return self.llm_provider.generate_response(query, context, chat_history, route)
