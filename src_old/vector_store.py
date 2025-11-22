import logging
import os
from typing import List, Tuple
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever

# Configuración de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_vector_db(chunks: List[Document], persist_directory: str = "./faiss_index") -> Tuple[FAISS, BM25Retriever]:
    """
    Crea o carga una base de datos vectorial (FAISS) y un retriever BM25.
    Implementa persistencia para evitar re-indexar.

    Args:
        chunks (List[Document]): Lista de documentos fragmentados.
        persist_directory (str): Directorio donde guardar/cargar el índice.

    Returns:
        Tuple[FAISS, BM25Retriever]: Tupla con el VectorStore y el BM25Retriever.
    """
    try:
        # Cambio a modelo multilingüe para mejor soporte en español
        model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        logger.info(f"Inicializando modelo de embeddings ({model_name})...")
        
        # Configuración para correr en CPU explícitamente
        model_kwargs = {'device': 'cpu'}
        encode_kwargs = {'normalize_embeddings': False}
        
        embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs
        )

        vector_store = None
        
        # Intentar cargar índice existente si no hay nuevos chunks (o si se prefiere persistencia)
        # Nota: Para este MVP, si se pasan chunks, re-creamos el índice. 
        # En un sistema real, verificaríamos si los chunks cambiaron.
        # Aquí asumimos: si hay chunks, re-indexamos. Si chunks es None/vacío, intentamos cargar.
        
        if chunks:
            logger.info(f"Creando índice vectorial con {len(chunks)} fragmentos...")
            vector_store = FAISS.from_documents(chunks, embeddings)
            logger.info(f"Guardando índice en {persist_directory}...")
            vector_store.save_local(persist_directory)
            
            logger.info("Creando BM25 Retriever...")
            bm25_retriever = BM25Retriever.from_documents(chunks)
            
        else:
            if os.path.exists(persist_directory):
                logger.info(f"Cargando índice existente desde {persist_directory}...")
                vector_store = FAISS.load_local(persist_directory, embeddings, allow_dangerous_deserialization=True)
                
                # Reconstruir BM25 desde los documentos del vector store
                logger.info("Reconstruyendo BM25 desde índice cargado...")
                # Extraer documentos del docstore de FAISS
                stored_docs = list(vector_store.docstore._dict.values())
                bm25_retriever = BM25Retriever.from_documents(stored_docs)
            else:
                raise ValueError("No se proporcionaron chunks y no existe índice guardado.")

        # Configurar k por defecto para BM25
        bm25_retriever.k = 5
        
        logger.info("Base de datos vectorial y BM25 listos.")
        return vector_store, bm25_retriever

    except Exception as e:
        logger.error(f"Error crítico creando/cargando el Vector Store: {e}")
        raise e
