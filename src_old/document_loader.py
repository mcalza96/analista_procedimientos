import logging
import os
from typing import List
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# Configuración básica de logging para seguimiento
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def process_documents(pdf_paths: List[str]) -> List[Document]:
    """
    Procesa una lista de documentos PDF: los carga y los divide en fragmentos (chunks).

    Esta función utiliza PyPDFLoader para la extracción de texto y RecursiveCharacterTextSplitter
    para la segmentación, optimizada para documentos técnicos.

    Args:
        pdf_paths (List[str]): Lista de rutas de archivos PDF a procesar.

    Returns:
        List[Document]: Lista de documentos divididos (chunks) listos para ser indexados.
                        Cada chunk conserva los metadatos originales (source, page).
    """
    all_chunks: List[Document] = []

    # Configuración del splitter
    # chunk_size=1000 y chunk_overlap=200 son valores empíricos buenos para contexto técnico
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""]  # Prioridad de separación estándar
    )

    for pdf_path in pdf_paths:
        try:
            logger.info(f"Iniciando carga de: {pdf_path}")
            
            # Cargar el PDF
            loader = PyPDFLoader(pdf_path)
            raw_documents = loader.load()
            
            # Dividir en chunks manteniendo metadatos
            file_chunks = text_splitter.split_documents(raw_documents)
            
            # Añadir metadato 'source_file' con solo el nombre del archivo
            filename = os.path.basename(pdf_path)
            for chunk in file_chunks:
                chunk.metadata['source_file'] = filename
            
            # Acumular resultados
            all_chunks.extend(file_chunks)
            
            logger.info(f"Procesado exitosamente: {pdf_path} - {len(file_chunks)} chunks generados.")

        except Exception as e:
            logger.error(f"Error crítico procesando {pdf_path}: {e}")
            # Continuamos con el siguiente archivo a pesar del error
            continue

    return all_chunks
