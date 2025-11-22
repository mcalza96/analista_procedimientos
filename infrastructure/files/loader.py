import logging
import os
from typing import List, Any
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from llama_parse import LlamaParse
from config.settings import settings
from core.interfaces.document_loader import DocumentLoaderRepository

logger = logging.getLogger(__name__)

class DocumentLoader(DocumentLoaderRepository):
    def load_documents(self, pdf_paths: List[str]) -> List[Document]:
        all_chunks: List[Document] = []
        
        # Configurar splitter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", " ", ""]
        )

        # Inicializar LlamaParse si hay API Key
        parser = None
        if settings.LLAMA_CLOUD_API_KEY:
            try:
                parser = LlamaParse(
                    api_key=settings.LLAMA_CLOUD_API_KEY,
                    result_type="markdown",
                    verbose=True
                )
            except Exception as e:
                logger.warning(f"Error al inicializar LlamaParse: {e}")

        for pdf_path in pdf_paths:
            try:
                logger.info(f"Iniciando carga de: {pdf_path}")
                raw_documents = []
                filename = os.path.basename(pdf_path)
                used_parser = "PyPDFLoader"

                # Intentar usar LlamaParse primero
                if parser:
                    try:
                        logger.info(f"Intentando procesar {filename} con LlamaParse...")
                        llama_docs = parser.load_data(pdf_path)
                        
                        # Convertir documentos de LlamaIndex a LangChain
                        for doc in llama_docs:
                            # LlamaIndex usa .text, LangChain usa page_content
                            lc_doc = Document(
                                page_content=doc.text,
                                metadata={
                                    "source_file": filename,
                                    "parser": "LlamaParse"
                                }
                            )
                            raw_documents.append(lc_doc)
                        
                        if raw_documents:
                            used_parser = "LlamaParse"
                            
                    except Exception as e:
                        logger.warning(f"Fallo LlamaParse para {filename}: {e}. Usando fallback.")
                        raw_documents = []

                # Fallback a PyPDFLoader
                if not raw_documents:
                    logger.info(f"Procesando {filename} con PyPDFLoader...")
                    loader = PyPDFLoader(pdf_path)
                    raw_documents = loader.load()
                    
                    # Asegurar metadatos
                    for doc in raw_documents:
                        doc.metadata['source_file'] = filename
                        doc.metadata['parser'] = "PyPDFLoader"
                    used_parser = "PyPDFLoader"

                # Dividir documentos
                if raw_documents:
                    file_chunks = text_splitter.split_documents(raw_documents)
                    all_chunks.extend(file_chunks)
                    logger.info(f"Procesado exitosamente: {pdf_path} ({used_parser}) - {len(file_chunks)} chunks generados.")
                else:
                    logger.warning(f"No se pudo extraer contenido de {pdf_path}")

            except Exception as e:
                logger.error(f"Error crítico procesando {pdf_path}: {e}")
                continue

        return all_chunks
