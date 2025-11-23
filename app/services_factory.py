from infrastructure.llm.groq_provider import GroqProvider
from infrastructure.vector_store.faiss_repository import FAISSRepository
from infrastructure.files.loader import DocumentLoader
from infrastructure.ai.semantic_router import SemanticRouter
from infrastructure.storage.session_manager import FileSessionRepository
from infrastructure.storage.local_file_storage import LocalFileStorage
from infrastructure.logging.feedback_logger import FeedbackLogger
from core.services.chat_service import ChatService
from core.services.document_service import DocumentService
from core.services.prompt_manager import PromptManager

class ServicesFactory:
    @staticmethod
    def create_services():
        """Creates and returns a dictionary of initialized services."""
        llm_provider = GroqProvider()
        vector_repo = FAISSRepository()
        doc_loader = DocumentLoader()
        router_repo = SemanticRouter()
        session_repo = FileSessionRepository()
        file_storage = LocalFileStorage()
        feedback_logger = FeedbackLogger()
        doc_service = DocumentService(doc_loader, file_storage)
        prompt_manager = PromptManager()
        
        return {
            "llm_provider": llm_provider,
            "vector_repo": vector_repo,
            "doc_loader": doc_loader,
            "router_repo": router_repo,
            "session_manager": session_repo,
            "file_storage": file_storage,
            "feedback_logger": feedback_logger,
            "doc_service": doc_service,
            "prompt_manager": prompt_manager
        }

    @staticmethod
    def create_chat_service(llm_provider, vector_repo, doc_loader, router_repo, prompt_manager):
        """Creates a ChatService instance."""
        return ChatService(
            llm_provider=llm_provider,
            vector_store_repo=vector_repo,
            document_loader=doc_loader,
            router_repo=router_repo,
            prompt_manager=prompt_manager
        )
