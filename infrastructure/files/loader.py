import logging
import os
from typing import List, Any, Optional
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from llama_parse import LlamaParse
from config.settings import settings
from core.interfaces.document_loader import DocumentLoaderRepository

logger = logging.getLogger(__name__)

class DocumentLoader(DocumentLoaderRepository):
    def __init__(self):
        self.parser = self._initialize_llama_parse()

    def _initialize_llama_parse(self) -> Optional[LlamaParse]:
        """Inicializa LlamaParse si hay API Key disponible."""
        if settings.LLAMA_CLOUD_API_KEY:
            try:
                return LlamaParse(
                    api_key=settings.LLAMA_CLOUD_API_KEY,
                    result_type="markdown",
                    verbose=True
                )
            except Exception as e:
                logger.warning(f"Error al inicializar LlamaParse: {e}")
        return None

    def _load_with_llama_parse(self, pdf_path: str, filename: str) -> List[Document]:
        """Intenta cargar documentos usando LlamaParse."""
        if not self.parser:
            return []
            
        try:
            logger.info(f"Intentando procesar {filename} con LlamaParse...")
            llama_docs = self.parser.load_data(pdf_path)
            
            raw_documents = []
            for doc in llama_docs:
                lc_doc = Document(
                    page_content=doc.text,
                    metadata={
                        "source_file": filename,
                        "source": filename,
                        "parser": "LlamaParse"
                    }
                )
                raw_documents.append(lc_doc)
            return raw_documents
        except Exception as e:
            logger.warning(f"Fallo LlamaParse para {filename}: {e}. Usando fallback.")
            return []

    def _load_with_pypdf(self, pdf_path: str, filename: str) -> List[Document]:
        """Carga documentos usando PyPDFLoader como fallback."""
        try:
            logger.info(f"Procesando {filename} con PyPDFLoader...")
            loader = PyPDFLoader(pdf_path)
            raw_documents = loader.load()
            
            for doc in raw_documents:
                doc.metadata['source_file'] = filename
                doc.metadata['source'] = filename
                doc.metadata['parser'] = "PyPDFLoader"
            return raw_documents
        except Exception as e:
            logger.error(f"Error en PyPDFLoader para {filename}: {e}")
            return []

    def load_documents(self, pdf_paths: List[str]) -> List[Document]:
        all_chunks: List[Document] = []
        
        for pdf_path in pdf_paths:
            try:
                logger.info(f"Iniciando carga de: {pdf_path}")
                filename = os.path.basename(pdf_path)
                
                # Intentar LlamaParse
                raw_documents = self._load_with_llama_parse(pdf_path, filename)
                used_parser = "LlamaParse" if raw_documents else "PyPDFLoader"

                # Fallback a PyPDFLoader
                if not raw_documents:
                    raw_documents = self._load_with_pypdf(pdf_path, filename)

                if raw_documents:
                    all_chunks.extend(raw_documents)
                    logger.info(f"Procesado exitosamente: {pdf_path} ({used_parser}) - {len(raw_documents)} documentos padres generados.")
                else:
                    logger.warning(f"No se pudo extraer contenido de {pdf_path}")

            except Exception as e:
                logger.error(f"Error crítico procesando {pdf_path}: {e}")
                continue

        return all_chunks
