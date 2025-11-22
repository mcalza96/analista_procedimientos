import logging
import os
from typing import List, Tuple, Any
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
try:
    from langchain.retrievers import ParentDocumentRetriever
    from langchain.storage import LocalFileStore
    from langchain.storage._lc_store import create_kv_docstore
except ImportError:
    from langchain_classic.retrievers import ParentDocumentRetriever
    from langchain_classic.storage import LocalFileStore
    from langchain_classic.storage._lc_store import create_kv_docstore

from langchain_text_splitters import RecursiveCharacterTextSplitter
from core.interfaces.vector_store import VectorStoreRepository
from config.settings import settings
import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore

logger = logging.getLogger(__name__)

class FAISSRepository(VectorStoreRepository):
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(
            model_name=settings.EMBEDDING_MODEL,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': False}
        )

    def create_vector_db(self, documents: List[Document]) -> Tuple[Any, Any]:
        try:
            vector_store = None
            bm25_retriever = None
            
            # 1. Configurar Almacenamiento de Documentos (Padres)
            # Usamos una carpeta específica para el índice de documentos
            docstore_path = "./docstore_index"
            fs = LocalFileStore(docstore_path)
            store = create_kv_docstore(fs)
            
            # 2. Configurar Splitters Duales
            # child_splitter: Chunks pequeños para búsqueda vectorial precisa
            child_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
            # parent_splitter: Chunks grandes para contexto completo al LLM
            parent_splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=500)

            # 3. Inicializar o Cargar Vector Store (Hijos)
            if os.path.exists(settings.PERSIST_DIRECTORY):
                logger.info(f"Cargando índice existente desde {settings.PERSIST_DIRECTORY}...")
                vector_store = FAISS.load_local(
                    settings.PERSIST_DIRECTORY, 
                    self.embeddings, 
                    allow_dangerous_deserialization=True
                )
            else:
                logger.info("Inicializando nuevo índice FAISS vacío...")
                embedding_size = len(self.embeddings.embed_query("test"))
                index = faiss.IndexFlatL2(embedding_size)
                vector_store = FAISS(
                    embedding_function=self.embeddings,
                    index=index,
                    docstore=InMemoryDocstore(),
                    index_to_docstore_id={}
                )

            # 4. Configurar ParentDocumentRetriever
            retriever = ParentDocumentRetriever(
                vectorstore=vector_store,
                docstore=store,
                child_splitter=child_splitter,
                parent_splitter=parent_splitter,
            )

            # 5. Agregar documentos si se proporcionan
            if documents:
                logger.info(f"Agregando {len(documents)} documentos al ParentDocumentRetriever...")
                # Esto automáticamente:
                # 1. Divide docs en padres (2000 chars)
                # 2. Guarda padres en docstore
                # 3. Divide padres en hijos (400 chars)
                # 4. Indexa hijos en vectorstore
                retriever.add_documents(documents, ids=None)
                
                logger.info(f"Guardando índice vectorial en {settings.PERSIST_DIRECTORY}...")
                vector_store.save_local(settings.PERSIST_DIRECTORY)
                
                # Crear BM25 con los documentos padres
                # Necesitamos generar los mismos chunks padres que generó el retriever
                logger.info("Generando chunks padres para BM25...")
                parent_docs_for_bm25 = parent_splitter.split_documents(documents)
                bm25_retriever = BM25Retriever.from_documents(parent_docs_for_bm25)
            else:
                # Reconstruir BM25 desde el DocStore
                logger.info("Reconstruyendo BM25 desde DocStore...")
                stored_docs = []
                # Iterar sobre todos los documentos en el store
                # yield_keys devuelve las claves de los documentos almacenados
                for key in store.yield_keys():
                    doc = store.mget([key])[0]
                    if doc:
                        stored_docs.append(doc)
                
                if stored_docs:
                    bm25_retriever = BM25Retriever.from_documents(stored_docs)
                else:
                    logger.warning("No se encontraron documentos en el DocStore para BM25.")
                    # Aún devolvemos el retriever aunque BM25 falle, para no romper todo
            
            if bm25_retriever:
                bm25_retriever.k = 5
            
            # Devolvemos el retriever (ParentDocumentRetriever) y el bm25
            return retriever, bm25_retriever

        except Exception as e:
            logger.error(f"Error crítico creando/cargando el Vector Store: {e}")
            raise e

    def get_retriever(self, vectorstore: Any, bm25_retriever: BM25Retriever):
        # This method might be redundant if we return the retrievers directly, 
        # but useful if we want to encapsulate the Ensemble creation here.
        # For now, we'll keep it simple and let the service handle Ensemble or move it here.
        pass
