import os
import shutil
from typing import List, Any, Tuple, Optional
from core.interfaces.document_loader import DocumentLoaderRepository
from core.interfaces.vector_store import VectorStoreRepository

class DocumentService:
    """
    Servicio encargado de la gestión, procesamiento e ingesta de documentos.
    """
    def __init__(self, doc_loader: DocumentLoaderRepository) -> None:
        self.doc_loader = doc_loader

    def process_and_ingest_files(
        self, 
        uploaded_files: List[Any], 
        session_path: str, 
        vector_repo: VectorStoreRepository
    ) -> Tuple[Optional[Any], Optional[Any], int]:
        """
        Procesa archivos subidos, los carga y actualiza el repositorio vectorial.
        
        Args:
            uploaded_files: Lista de archivos subidos (objetos tipo Streamlit UploadedFile).
            session_path: Ruta del directorio de la sesión actual.
            vector_repo: Repositorio vectorial para actualizar el índice.
            
        Returns:
            Tuple[Optional[Any], Optional[Any], int]: 
                - Retriever actualizado (o None si falla).
                - BM25 Retriever actualizado (o None si falla).
                - Número de chunks procesados.
        """
        temp_dir = os.path.join(session_path, "temp_uploads")
        os.makedirs(temp_dir, exist_ok=True)
        
        file_paths: List[str] = []
        try:
            for uploaded_file in uploaded_files:
                file_path = os.path.join(temp_dir, uploaded_file.name)
                # Asumimos que uploaded_file tiene getbuffer() (Streamlit) o read()
                with open(file_path, "wb") as f:
                    if hasattr(uploaded_file, 'getbuffer'):
                        f.write(uploaded_file.getbuffer())
                    else:
                        f.write(uploaded_file.read())
                file_paths.append(file_path)
            
            chunks = self.doc_loader.load_documents(file_paths)
            
            if chunks:
                new_retriever, new_bm25 = vector_repo.add_documents(session_path, chunks)
                return new_retriever, new_bm25, len(chunks)
            else:
                return None, None, 0
                
        finally:
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
