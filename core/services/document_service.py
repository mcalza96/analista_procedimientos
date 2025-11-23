import logging
from datetime import datetime
from typing import List, Any, Tuple, Optional
from langchain_core.documents import Document
from core.interfaces.document_loader import DocumentLoaderRepository
from core.interfaces.vector_store import VectorStoreRepository
from core.interfaces.file_storage import FileStorageRepository

logger = logging.getLogger(__name__)

class DocumentService:
    """
    Servicio encargado de la gestión, procesamiento e ingesta de documentos.
    """
    def __init__(self, doc_loader: DocumentLoaderRepository, file_storage: FileStorageRepository) -> None:
        self.doc_loader = doc_loader
        self.file_storage = file_storage

    def ingest_text_as_document(
        self, 
        text_content: str, 
        title: str, 
        session_path: str, 
        vector_repo: VectorStoreRepository
    ) -> bool:
        """
        Ingesta texto directamente como un documento en la base de conocimiento.
        
        Args:
            text_content: El contenido del texto a guardar.
            title: Título para el documento.
            session_path: Ruta de la sesión.
            vector_repo: Repositorio vectorial.
            
        Returns:
            bool: True si fue exitoso, False si falló.
        """
        try:
            doc = Document(
                page_content=text_content,
                metadata={
                    "source": "user_note",
                    "title": title,
                    "created_at": datetime.now().isoformat(),
                    "type": "qa_insight"
                }
            )
            
            vector_repo.add_documents(session_path, [doc])
            return True
        except Exception as e:
            logger.error(f"Error ingesting text document: {e}")
            return False

    def process_and_ingest_files(
        self, 
        uploaded_files: List[Any], 
        session_path: str, 
        vector_repo: VectorStoreRepository
    ) -> Tuple[Optional[Any], Optional[Any], int]:
        """
        Procesa archivos subidos, los guarda permanentemente y actualiza el repositorio vectorial.
        """
        file_paths: List[str] = []
        
        try:
            for uploaded_file in uploaded_files:
                # Usar el repositorio de almacenamiento para guardar el archivo
                file_path = self.file_storage.save_file(session_path, uploaded_file.name, uploaded_file)
                file_paths.append(file_path)
            
            if not file_paths:
                return None, None, 0

            # Cargar y procesar documentos
            chunks = self.doc_loader.load_documents(file_paths)
            
            if chunks:
                new_retriever, new_bm25 = vector_repo.add_documents(session_path, chunks)
                return new_retriever, new_bm25, len(chunks)
            else:
                return None, None, 0
                
        except Exception as e:
            logger.error(f"Error processing files: {e}")
            return None, None, 0

    def list_files(self, session_path: str) -> List[str]:
        """Lista los archivos fuente almacenados en la sesión."""
        return self.file_storage.list_files(session_path)

    def delete_file(self, session_path: str, filename: str, vector_repo: VectorStoreRepository) -> bool:
        """
        Elimina un archivo y reconstruye el índice vectorial.
        Estrategia: Borrado físico + Reconstrucción total (Clean Rebuild).
        """
        try:
            # 1. Borrar archivo físico usando el repositorio
            if not self.file_storage.delete_file(session_path, filename):
                 logger.warning(f"Advertencia: El archivo {filename} no se pudo borrar o no existía, pero se procederá a limpiar el índice.")
            
            # 2. Limpiar índices vectoriales (Delegado al repositorio)
            if not vector_repo.clear_index(session_path):
                logger.error("Error al limpiar el índice vectorial.")
                return False
            
            # 3. Re-indexar todos los archivos restantes
            remaining_filenames = self.file_storage.list_files(session_path)
            
            if not remaining_filenames:
                # Si no quedan archivos, inicializar DB vacía
                vector_repo.get_vector_db(session_path)
                return True
                
            # Obtener rutas completas
            remaining_files_paths = [self.file_storage.get_file_path(session_path, f) for f in remaining_filenames]

            # Cargar y procesar documentos restantes
            chunks = self.doc_loader.load_documents(remaining_files_paths)
            if chunks:
                vector_repo.add_documents(session_path, chunks)
                
            return True
        except Exception as e:
            logger.error(f"Error eliminando archivo {filename}: {e}")
            return False
