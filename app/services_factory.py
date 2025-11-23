import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings
from config.settings import settings
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
    @st.cache_resource(show_spinner="Cargando modelos de IA...")
    def get_embedding_model():
        """
        Carga y cachea el modelo de embeddings.
        Esto evita recargar el modelo pesado en cada recarga de página.
        """
        return HuggingFaceEmbeddings(
            model_name=settings.EMBEDDING_MODEL,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': False}
        )

    @staticmethod
    @st.cache_resource(show_spinner="Iniciando servicios del sistema...")
    def create_services():
        """Creates and returns a dictionary of initialized services."""
        llm_provider = GroqProvider()
        
        # Inyectar modelo cacheado
        embeddings = ServicesFactory.get_embedding_model()
        vector_repo = FAISSRepository(embeddings)
        
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
