import logging
import os
import shutil
from typing import List, Tuple, Any, Optional
from pathlib import Path

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import ParentDocumentRetriever
from langchain_classic.storage import LocalFileStore
from langchain_classic.storage._lc_store import create_kv_docstore

from langchain_text_splitters import RecursiveCharacterTextSplitter
from core.interfaces.vector_store import VectorStoreRepository
from config.settings import settings
from infrastructure.constants import DIR_DOC_STORE, DIR_VECTOR_STORE, FILE_FAISS_INDEX
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

    def _get_splitters(self) -> Tuple[RecursiveCharacterTextSplitter, RecursiveCharacterTextSplitter]:
        """Configura y retorna los splitters para documentos hijos y padres."""
        child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.CHUNK_SIZE_CHILD, 
            chunk_overlap=settings.CHUNK_OVERLAP_CHILD
        )
        parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.CHUNK_SIZE_PARENT, 
            chunk_overlap=settings.CHUNK_OVERLAP_PARENT
        )
        return child_splitter, parent_splitter

    def _load_or_create_vector_store(self, vectorstore_path: Path) -> FAISS:
        """Carga un índice FAISS existente o crea uno nuevo."""
        if vectorstore_path.exists() and (vectorstore_path / FILE_FAISS_INDEX).exists():
            logger.info(f"Cargando índice FAISS existente desde {vectorstore_path}...")
            return FAISS.load_local(
                str(vectorstore_path), 
                self.embeddings, 
                allow_dangerous_deserialization=True
            )
        
        logger.info("Inicializando nuevo índice FAISS vacío...")
        embedding_size = len(self.embeddings.embed_query("test"))
        index = faiss.IndexFlatL2(embedding_size)
        return FAISS(
            embedding_function=self.embeddings,
            index=index,
            docstore=InMemoryDocstore(),
            index_to_docstore_id={}
        )

    def get_vector_db(self, session_path: str) -> Tuple[Any, Any]:
        """
        Obtiene (carga o crea) la base de datos vectorial para una sesión.
        """
        session_dir = Path(session_path)
        if not session_dir.exists():
            raise ValueError(f"El directorio de sesión no existe: {session_path}")

        docstore_path = session_dir / DIR_DOC_STORE
        vectorstore_path = session_dir / DIR_VECTOR_STORE

        try:
            # 1. Configurar Almacenamiento de Documentos (Padres)
            fs = LocalFileStore(str(docstore_path))
            store = create_kv_docstore(fs)

            # 2. Configurar Splitters
            child_splitter, parent_splitter = self._get_splitters()

            # 3. Cargar o Inicializar Vector Store (FAISS)
            vector_store = self._load_or_create_vector_store(vectorstore_path)

            # 4. Configurar ParentDocumentRetriever
            retriever = ParentDocumentRetriever(
                vectorstore=vector_store,
                docstore=store,
                child_splitter=child_splitter,
                parent_splitter=parent_splitter,
                search_kwargs={"k": settings.RETRIEVER_K_PARENT}
            )

            # 5. Configurar BM25 Retriever
            bm25_retriever = self._create_bm25_retriever(store)

            return retriever, bm25_retriever

        except Exception as e:
            logger.error(f"Error crítico obteniendo Vector DB para sesión {session_path}: {e}")
            raise RuntimeError(f"No se pudo inicializar la base de datos vectorial: {e}")

    def add_documents(self, session_path: str, new_documents: List[Document]) -> Tuple[Any, Any]:
        """
        Agrega nuevos documentos a la sesión existente.
        """
        try:
            # Obtener componentes actuales
            retriever, _ = self.get_vector_db(session_path)
            
            if new_documents:
                logger.info(f"Agregando {len(new_documents)} documentos a sesión {session_path}...")
                retriever.add_documents(new_documents, ids=None)
                
                # Persistir Vector Store
                vectorstore_path = Path(session_path) / DIR_VECTOR_STORE
                logger.info(f"Guardando índice vectorial actualizado en {vectorstore_path}...")
                retriever.vectorstore.save_local(str(vectorstore_path))
                
                # Reconstruir BM25 con los nuevos datos
                docstore_path = Path(session_path) / DIR_DOC_STORE
                fs = LocalFileStore(str(docstore_path))
                store = create_kv_docstore(fs)
                bm25_retriever = self._create_bm25_retriever(store)
            else:
                bm25_retriever = self._create_bm25_retriever(retriever.docstore)

            return retriever, bm25_retriever

        except Exception as e:
            logger.error(f"Error agregando documentos a sesión {session_path}: {e}")
            raise e

    def _create_bm25_retriever(self, store: Any) -> Optional[BM25Retriever]:
        """Helper para crear BM25 desde el docstore."""
        try:
            stored_docs = []
            for key in store.yield_keys():
                doc = store.mget([key])[0]
                if doc:
                    stored_docs.append(doc)
            
            if stored_docs:
                bm25 = BM25Retriever.from_documents(stored_docs)
                bm25.k = settings.RETRIEVER_K_BM25
                return bm25
            return None
        except Exception as e:
            logger.warning(f"No se pudo crear BM25Retriever: {e}")
            return None

    def clear_index(self, session_path: str) -> bool:
        """
        Elimina físicamente los directorios del índice vectorial y docstore.
        """
        try:
            session_dir = Path(session_path)
            vectorstore_path = session_dir / DIR_VECTOR_STORE
            docstore_path = session_dir / DIR_DOC_STORE

            if vectorstore_path.exists():
                shutil.rmtree(vectorstore_path)
                logger.info(f"Eliminado índice vectorial en {vectorstore_path}")
            
            if docstore_path.exists():
                shutil.rmtree(docstore_path)
                logger.info(f"Eliminado docstore en {docstore_path}")

            # Recrear directorios vacíos
            vectorstore_path.mkdir(parents=True, exist_ok=True)
            docstore_path.mkdir(parents=True, exist_ok=True)
            
            return True
        except Exception as e:
            logger.error(f"Error limpiando índice para sesión {session_path}: {e}")
            return False
