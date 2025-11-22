import logging
import os
from typing import List, Tuple, Any, Optional
from pathlib import Path

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
# Updated imports for langchain compatibility
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

    def get_vector_db(self, session_path: str) -> Tuple[Any, Any]:
        """
        Obtiene (carga o crea) la base de datos vectorial para una sesión.
        
        Args:
            session_path: Ruta al directorio de la sesión.
            
        Returns:
            Tuple[ParentDocumentRetriever, BM25Retriever]: Los retrievers configurados.
            
        Raises:
            ValueError: Si el directorio de sesión no existe.
            RuntimeError: Si hay un error crítico cargando el índice.
        """
        session_dir = Path(session_path)
        if not session_dir.exists():
            raise ValueError(f"El directorio de sesión no existe: {session_path}")

        docstore_path = session_dir / "doc_store"
        vectorstore_path = session_dir / "vector_store"

        try:
            # 1. Configurar Almacenamiento de Documentos (Padres)
            fs = LocalFileStore(str(docstore_path))
            store = create_kv_docstore(fs)

            # 2. Configurar Splitters
            child_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
            parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)

            # 3. Cargar o Inicializar Vector Store (FAISS)
            if vectorstore_path.exists() and (vectorstore_path / "index.faiss").exists():
                logger.info(f"Cargando índice FAISS existente desde {vectorstore_path}...")
                # ADVERTENCIA DE SEGURIDAD: allow_dangerous_deserialization=True es necesario para cargar
                # archivos pickle locales generados por FAISS. Solo debe usarse con índices confiables
                # generados internamente por esta misma aplicación.
                vector_store = FAISS.load_local(
                    str(vectorstore_path), 
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
            # Aumentamos search_kwargs k=60 para traer más candidatos iniciales
            # Esto es crucial para preguntas complejas que requieren datos de múltiples secciones
            retriever = ParentDocumentRetriever(
                vectorstore=vector_store,
                docstore=store,
                child_splitter=child_splitter,
                parent_splitter=parent_splitter,
                search_kwargs={"k": 60}
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
        
        Args:
            session_path: Ruta de la sesión.
            new_documents: Lista de documentos a agregar.
            
        Returns:
            Tuple[ParentDocumentRetriever, BM25Retriever]: Retrievers actualizados.
        """
        try:
            # Obtener componentes actuales
            retriever, _ = self.get_vector_db(session_path)
            
            if new_documents:
                logger.info(f"Agregando {len(new_documents)} documentos a sesión {session_path}...")
                retriever.add_documents(new_documents, ids=None)
                
                # Persistir Vector Store
                vectorstore_path = Path(session_path) / "vector_store"
                logger.info(f"Guardando índice vectorial actualizado en {vectorstore_path}...")
                retriever.vectorstore.save_local(str(vectorstore_path))
                
                # Reconstruir BM25 con los nuevos datos
                docstore_path = Path(session_path) / "doc_store"
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
                bm25.k = 30  # Aumentamos de 10 a 30 para capturar más palabras clave
                return bm25
            return None
        except Exception as e:
            logger.warning(f"No se pudo crear BM25Retriever: {e}")
            return None
