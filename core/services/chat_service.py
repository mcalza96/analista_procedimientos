from typing import List, Any, Tuple, Optional, Generator
from langchain_classic.retrievers import EnsembleRetriever
# from sentence_transformers import CrossEncoder
from langchain_core.documents import Document
from core.interfaces.llm_provider import LLMProvider
from core.interfaces.vector_store import VectorStoreRepository
from core.interfaces.document_loader import DocumentLoaderRepository
from core.interfaces.router import RouterRepository
from core.domain.models import ChatResponse, SourceDocument
from core.services.prompt_manager import PromptManager
import logging

logger = logging.getLogger(__name__)

class ChatService:
    """
    Servicio principal de Chat que orquesta la recuperación de información,
    el reranking y la generación de respuestas.
    """
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
        # Desactivamos Reranker por problemas con idioma español en modelo actual
        # self.reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

    def process_uploaded_files(self, file_paths: List[str]) -> None:
        """Procesa archivos subidos y crea la base de datos vectorial."""
        chunks = self.document_loader.load_documents(file_paths)
        self.vector_store, self.bm25_retriever = self.vector_store_repo.create_vector_db(chunks)

    def load_existing_db(self) -> None:
        """Carga una base de datos vectorial existente vacía o por defecto."""
        self.vector_store, self.bm25_retriever = self.vector_store_repo.create_vector_db([])

    def _rerank_documents(self, query: str, docs: List[Document]) -> List[Document]:
        """
        Passthrough (Reranker deshabilitado).
        """
        if not docs:
            return []
            
        # Retornamos los top 15 documentos del Ensemble directamente
        # El Ensemble ya combina BM25 (palabras clave) y Vector (semántico)
        return docs[:15]

    def _retrieve_documents(self, query: str) -> Tuple[List[SourceDocument], str]:
        """
        Recupera y reordena documentos relevantes para la consulta.
        
        Returns:
            Tuple[List[SourceDocument], str]: Lista de documentos fuente y el string de contexto formateado.
        """
        if not self.vector_store or not self.bm25_retriever:
            return [], ""

        # Ensemble Retriever
        ensemble_retriever = EnsembleRetriever(
            retrievers=[self.bm25_retriever, self.vector_store],
            weights=[0.4, 0.6]
        )
        
        initial_docs = ensemble_retriever.invoke(query)
        
        # Reranking Avanzado
        top_docs = self._rerank_documents(query, initial_docs)
        
        # Prepare context string
        context_parts = []
        source_docs = []
        for i, doc in enumerate(top_docs):
            source_file = doc.metadata.get('source_file', 'unknown')
            page = doc.metadata.get('page', 0)
            # Add numbering to context so LLM can cite [1], [2], etc.
            context_parts.append(f"Document [{i+1}]\nContent: {doc.page_content}\nSource: {source_file}")
            
            source_docs.append(SourceDocument(
                page_content=doc.page_content,
                metadata=doc.metadata,
                source_file=source_file,
                page_number=page
            ))
            
        context_str = "\n\n".join(context_parts)
        return source_docs, context_str

    def get_response(self, query: str, chat_history: List[Any], route: str = None) -> ChatResponse:
        """
        Genera una respuesta a la consulta del usuario orquestando todo el flujo RAG.
        """
        try:
            # Paso 1: Routing
            if route is None:
                route = self.router_repo.route_query(query)
                
            # Paso 2: Retrieval & Logic based on route
            if route == "CHAT":
                prompt = PromptManager.get_chat_prompt(query)
                response_text = self.llm_provider.generate_response(prompt)
                return ChatResponse(answer=response_text, route=route)
                
            # For other routes (ANALYSIS, WALKTHROUGH, PRECISION), we need context
            source_docs, context_str = self._retrieve_documents(query)
            
            if not context_str:
                 return ChatResponse(answer="Por favor, carga documentos primero.", route="ERROR")

            # Paso 3: Prompting
            if route == "ANALYSIS":
                prompt = PromptManager.get_audit_prompt(context_str)
            elif route == "WALKTHROUGH":
                prompt = PromptManager.get_walkthrough_prompt(context_str)
            else: # PRECISION
                prompt = PromptManager.get_precision_prompt(context_str)
                
            full_prompt = f"{prompt}\n\nPregunta: {query}"

            # Paso 4: Generation
            response_text = self.llm_provider.generate_response(full_prompt)

            # Paso 5: Return ChatResponse
            return ChatResponse(
                answer=response_text,
                source_documents=source_docs,
                route=route
            )
            
        except Exception as e:
            logger.error(f"Error en ChatService.get_response: {e}")
            return ChatResponse(answer=f"Ocurrió un error procesando tu solicitud: {str(e)}", route="ERROR")

    def get_streaming_response(self, query: str, chat_history: List[Any], route: str = None) -> Tuple[Generator[str, None, None], List[SourceDocument], str]:
        """
        Genera una respuesta en streaming a la consulta del usuario.
        
        Returns:
            Tuple[Generator, List[SourceDocument], str]: Generador de texto, documentos fuente y ruta.
        """
        try:
            # Paso 1: Routing
            if route is None:
                route = self.router_repo.route_query(query)
                
            # Paso 2: Retrieval & Logic based on route
            if route == "CHAT":
                prompt = PromptManager.get_chat_prompt(query)
                generator = self.llm_provider.generate_stream(prompt)
                return generator, [], route
                
            # For other routes (ANALYSIS, WALKTHROUGH, PRECISION), we need context
            source_docs, context_str = self._retrieve_documents(query)
            
            if not context_str:
                 def error_gen(): yield "Por favor, carga documentos primero."
                 return error_gen(), [], "ERROR"

            # Paso 3: Prompting
            if route == "ANALYSIS":
                prompt = PromptManager.get_audit_prompt(context_str)
            elif route == "WALKTHROUGH":
                prompt = PromptManager.get_walkthrough_prompt(context_str)
            else: # PRECISION
                prompt = PromptManager.get_precision_prompt(context_str)
                
            full_prompt = f"{prompt}\n\nPregunta: {query}"

            # Paso 4: Generation (Stream)
            generator = self.llm_provider.generate_stream(full_prompt)

            # Paso 5: Return Generator and Sources
            return generator, source_docs, route
            
        except Exception as e:
            logger.error(f"Error en ChatService.get_streaming_response: {e}")
            def error_gen(): yield f"Ocurrió un error procesando tu solicitud: {str(e)}"
            return error_gen(), [], "ERROR"

