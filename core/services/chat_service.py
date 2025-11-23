from typing import List, Any, Tuple, Optional, Generator
from langchain_classic.retrievers import EnsembleRetriever
from sentence_transformers import CrossEncoder
from langchain_core.documents import Document
from core.interfaces.llm_provider import LLMProvider
from core.interfaces.vector_store import VectorStoreRepository
from core.interfaces.document_loader import DocumentLoaderRepository
from core.interfaces.router import RouterRepository
from core.domain.models import ChatResponse, SourceDocument, LLMProviderError, RouteType
from core.services.prompt_manager import PromptManager
from config.settings import settings
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
        router_repo: RouterRepository,
        prompt_manager: PromptManager
    ):
        self.llm_provider = llm_provider
        self.vector_store_repo = vector_store_repo
        self.document_loader = document_loader
        self.router_repo = router_repo
        self.prompt_manager = prompt_manager
        self.vector_store = None
        self.bm25_retriever = None
        # Inicializamos Reranker Multilingüe Ligero
        try:
            self.reranker = CrossEncoder(settings.RERANKER_MODEL)
        except Exception as e:
            logger.warning(f"Error cargando reranker: {e}")
            self.reranker = None

    def _rerank_documents(self, query: str, docs: List[Document]) -> List[Document]:
        """
        Reordena los documentos recuperados usando un CrossEncoder Multilingüe.
        
        Args:
            query: La consulta del usuario.
            docs: Lista de documentos recuperados inicialmente.
            
        Returns:
            List[Document]: Top K documentos más relevantes reordenados.
        """
        if not docs:
            return []
            
        # Eliminar duplicados basados en contenido antes del reranking
        unique_docs = []
        seen_content = set()
        for doc in docs:
            if doc.page_content not in seen_content:
                unique_docs.append(doc)
                seen_content.add(doc.page_content)
        
        if not unique_docs or not self.reranker:
            return unique_docs[:settings.RERANKER_TOP_K]

        # Preparar pares para el CrossEncoder
        pairs = [[query, doc.page_content] for doc in unique_docs]
        
        # Predecir scores
        scores = self.reranker.predict(pairs)
        
        # Asignar scores a metadata y ordenar
        for doc, score in zip(unique_docs, scores):
            doc.metadata['score'] = float(score)

        # Ordenar por score descendente
        scored_docs = sorted(unique_docs, key=lambda x: x.metadata.get('score', 0), reverse=True)
        
        # Retornar top K
        return scored_docs[:settings.RERANKER_TOP_K]

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
            
            # Normalize route to Enum if it's a string (for backward compatibility or router output)
            try:
                route_enum = RouteType(route)
            except ValueError:
                route_enum = RouteType.CHAT # Default fallback

            # Paso 2: Retrieval & Logic based on route
            if route_enum == RouteType.CHAT:
                prompt = self.prompt_manager.get_chat_prompt(query)
                response_text = self.llm_provider.generate_response(prompt)
                return ChatResponse(answer=response_text, route=route_enum)
                
            # For other routes (ANALYSIS, WALKTHROUGH, PRECISION), we need context
            source_docs, context_str = self._retrieve_documents(query)
            
            if not context_str:
                 return ChatResponse(answer="Por favor, carga documentos primero.", route=RouteType.ERROR)

            # Paso 3: Prompting
            if route_enum == RouteType.ANALYSIS:
                prompt = self.prompt_manager.get_audit_prompt(context_str)
            elif route_enum == RouteType.WALKTHROUGH:
                prompt = self.prompt_manager.get_walkthrough_prompt(context_str)
            else: # PRECISION
                prompt = self.prompt_manager.get_precision_prompt(context_str)
                
            full_prompt = f"{prompt}\n\nPregunta: {query}"

            # Paso 4: Generation
            response_text = self.llm_provider.generate_response(full_prompt)

            # Paso 5: Return ChatResponse
            return ChatResponse(
                answer=response_text,
                source_documents=source_docs,
                route=route_enum
            )
        
        except LLMProviderError as e:
            logger.error(f"LLM Provider Error: {e}")
            return ChatResponse(answer="Lo siento, hubo un problema de comunicación con el modelo de IA. Por favor intenta de nuevo más tarde.", route=RouteType.ERROR)
            
        except Exception as e:
            logger.error(f"Error en ChatService.get_response: {e}")
            return ChatResponse(answer=f"Ocurrió un error procesando tu solicitud: {str(e)}", route=RouteType.ERROR)

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
            
            try:
                route_enum = RouteType(route)
            except ValueError:
                route_enum = RouteType.CHAT

            # Paso 2: Retrieval & Logic based on route
            if route_enum == RouteType.CHAT:
                prompt = self.prompt_manager.get_chat_prompt(query)
                generator = self.llm_provider.generate_stream(prompt)
                return generator, [], route_enum
                
            # For other routes (ANALYSIS, WALKTHROUGH, PRECISION), we need context
            source_docs, context_str = self._retrieve_documents(query)
            
            if not context_str:
                 def error_gen(): yield "Por favor, carga documentos primero."
                 return error_gen(), [], RouteType.ERROR

            # Paso 3: Prompting
            if route_enum == RouteType.ANALYSIS:
                prompt = self.prompt_manager.get_audit_prompt(context_str)
            elif route_enum == RouteType.WALKTHROUGH:
                prompt = self.prompt_manager.get_walkthrough_prompt(context_str)
            else: # PRECISION
                prompt = self.prompt_manager.get_precision_prompt(context_str)
                
            full_prompt = f"{prompt}\n\nPregunta: {query}"

            # Paso 4: Generation (Stream)
            generator = self.llm_provider.generate_stream(full_prompt)

            # Paso 5: Return Generator and Sources
            return generator, source_docs, route_enum
        
        except LLMProviderError as e:
            logger.error(f"LLM Provider Error: {e}")
            def error_gen(): yield "Lo siento, hubo un problema de comunicación con el modelo de IA. Por favor intenta de nuevo más tarde."
            return error_gen(), [], RouteType.ERROR
            
        except Exception as e:
            logger.error(f"Error en ChatService.get_streaming_response: {e}")
            def error_gen(): yield f"Ocurrió un error procesando tu solicitud: {str(e)}"
            return error_gen(), [], RouteType.ERROR
    
    def generate_context_summary(self) -> str:
        """
        Genera un resumen ejecutivo del contexto actual almacenado en la base vectorial.
        Utiliza una búsqueda amplia para obtener una muestra representativa del contenido.
        """
        if not self.vector_store:
            return "No hay contexto disponible para analizar. Por favor carga documentos primero."
            
        try:
            # 1. Recuperar una muestra amplia de documentos
            # Aumentamos k para tener más contexto y reducimos el riesgo de perder info clave
            # Usamos una query que busque estructura documental
            docs = self.vector_store.similarity_search(
                "objetivo alcance definiciones responsabilidades procedimiento", 
                k=15
            )
            
            if not docs:
                return "La base de conocimiento está vacía."
                
            # 2. Combinar contenido (Aumentamos límite de caracteres por chunk)
            # 1000 chars por chunk * 15 chunks = ~15k chars, manejable para modelos modernos
            context_text = "\n\n".join([
                f"--- Fragmento ({d.metadata.get('source_file', 'unknown')}) ---\n{d.page_content[:1000]}..." 
                for d in docs
            ])
            
            # 3. Prompt para el LLM
            prompt = self.prompt_manager.get_context_summary_prompt(context_text)
            
            return self.llm_provider.generate_response(prompt)
            
        except Exception as e:
            logger.error(f"Error generando resumen de contexto: {e}")
            return f"No se pudo generar el resumen del contexto debido a un error: {str(e)}"

