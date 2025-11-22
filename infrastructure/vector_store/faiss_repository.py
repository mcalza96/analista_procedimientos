import logging
import os
from typing import List, Tuple
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from core.interfaces.vector_store import VectorStoreRepository
from config.settings import settings

logger = logging.getLogger(__name__)

class FAISSRepository(VectorStoreRepository):
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(
            model_name=settings.EMBEDDING_MODEL,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': False}
        )

    def create_vector_db(self, chunks: List[Document]) -> Tuple[FAISS, BM25Retriever]:
        try:
            vector_store = None
            bm25_retriever = None
            
            if chunks:
                logger.info(f"Creando índice vectorial con {len(chunks)} fragmentos...")
                vector_store = FAISS.from_documents(chunks, self.embeddings)
                logger.info(f"Guardando índice en {settings.PERSIST_DIRECTORY}...")
                vector_store.save_local(settings.PERSIST_DIRECTORY)
                
                logger.info("Creando BM25 Retriever...")
                bm25_retriever = BM25Retriever.from_documents(chunks)
            else:
                if os.path.exists(settings.PERSIST_DIRECTORY):
                    logger.info(f"Cargando índice existente desde {settings.PERSIST_DIRECTORY}...")
                    vector_store = FAISS.load_local(settings.PERSIST_DIRECTORY, self.embeddings, allow_dangerous_deserialization=True)
                    
                    logger.info("Reconstruyendo BM25 desde índice cargado...")
                    stored_docs = list(vector_store.docstore._dict.values())
                    bm25_retriever = BM25Retriever.from_documents(stored_docs)
                else:
                    # Return None if no index exists and no chunks provided
                    return None, None

            if bm25_retriever:
                bm25_retriever.k = 5
            
            return vector_store, bm25_retriever

        except Exception as e:
            logger.error(f"Error crítico creando/cargando el Vector Store: {e}")
            raise e

    def get_retriever(self, vectorstore: FAISS, bm25_retriever: BM25Retriever):
        # This method might be redundant if we return the retrievers directly, 
        # but useful if we want to encapsulate the Ensemble creation here.
        # For now, we'll keep it simple and let the service handle Ensemble or move it here.
        pass
