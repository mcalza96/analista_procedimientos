import logging
import os
from typing import List
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

class DocumentLoader:
    @staticmethod
    def process_documents(pdf_paths: List[str]) -> List[Document]:
        all_chunks: List[Document] = []
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", " ", ""]
        )

        for pdf_path in pdf_paths:
            try:
                logger.info(f"Iniciando carga de: {pdf_path}")
                loader = PyPDFLoader(pdf_path)
                raw_documents = loader.load()
                file_chunks = text_splitter.split_documents(raw_documents)
                
                filename = os.path.basename(pdf_path)
                for chunk in file_chunks:
                    chunk.metadata['source_file'] = filename
                
                all_chunks.extend(file_chunks)
                logger.info(f"Procesado exitosamente: {pdf_path} - {len(file_chunks)} chunks generados.")

            except Exception as e:
                logger.error(f"Error crítico procesando {pdf_path}: {e}")
                continue

        return all_chunks
