from infrastructure.llm.groq_provider import GroqProvider
from infrastructure.vector_store.faiss_repository import FAISSRepository
from infrastructure.files.loader import DocumentLoader
from infrastructure.ai.semantic_router import SemanticRouter
from infrastructure.storage.session_manager import SessionManager
from core.services.chat_service import ChatService
from core.services.document_service import DocumentService

class ServicesFactory:
    @staticmethod
    def create_services():
        """Creates and returns a dictionary of initialized services."""
        llm_provider = GroqProvider()
        vector_repo = FAISSRepository()
        doc_loader = DocumentLoader()
        router_repo = SemanticRouter()
        session_manager = SessionManager()
        doc_service = DocumentService(doc_loader)
        
        return {
            "llm_provider": llm_provider,
            "vector_repo": vector_repo,
            "doc_loader": doc_loader,
            "router_repo": router_repo,
            "session_manager": session_manager,
            "doc_service": doc_service
        }

    @staticmethod
    def create_chat_service(llm_provider, vector_repo, doc_loader, router_repo):
        """Creates a ChatService instance."""
        return ChatService(
            llm_provider=llm_provider,
            vector_store_repo=vector_repo,
            document_loader=doc_loader,
            router_repo=router_repo
        )
