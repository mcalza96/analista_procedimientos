
import sys
import os
from pathlib import Path

# Add project root to path
sys.path.append(os.getcwd())

from infrastructure.vector_store.faiss_repository import FAISSRepository
from langchain_core.documents import Document

from sentence_transformers import CrossEncoder

def debug_retrieval():
    session_id = "562d752a-5117-4d49-afbd-9a8c5febda29"
    session_path = f"data/sessions/{session_id}"
    
    print(f"Checking session: {session_path}")
    
    repo = FAISSRepository()
    try:
        retriever, bm25 = repo.get_vector_db(session_path)
        
        query = "¿Cómo se reflejan los cambios estratégicos de desinversión de negocios en el Grupo Inditex (particularmente en Rusia, Argentina y Uruguay) en la estructura financiera individual de Industria de Diseño Textil, S.A. (Sociedad matriz) y cómo se contrasta esta evolución con su estrategia de inversión en activos no corrientes al cierre del ejercicio 2023?"
        print(f"\nQuerying: '{query[:50]}...'")
        
        # Simulate Ensemble
        vector_docs = retriever.invoke(query)
        bm25_docs = bm25.invoke(query) if bm25 else []
        
        # Combine unique docs
        seen = set()
        all_docs = []
        for d in vector_docs + bm25_docs:
            if d.page_content not in seen:
                seen.add(d.page_content)
                all_docs.append(d)
                
        print(f"Total unique docs retrieved: {len(all_docs)}")
        
        # Check if target docs are in the retrieved set
        found_target = False
        for i, d in enumerate(all_docs):
            if "183 millones" in d.page_content:
                print(f">>> TARGET DOC (183M) FOUND AT INDEX {i} IN INITIAL RETRIEVAL <<<")
                found_target = True
                break
        
        if not found_target:
            print(">>> TARGET DOC WAS NOT RETRIEVED <<<")

        # Rerank
        print("Reranking...")
        reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        pairs = [[query, doc.page_content] for doc in all_docs]
        scores = reranker.predict(pairs)
        
        scored_docs = sorted(zip(all_docs, scores), key=lambda x: x[1], reverse=True)
        
        print("\n--- Top 7 Reranked Docs ---")
        for i, (doc, score) in enumerate(scored_docs[:7]):
            print(f"\nRank {i+1} (Score: {score:.4f})")
            print(f"Source: {doc.metadata.get('source_file')}")
            print(f"Content: {doc.page_content[:300]}...")
            
        # Find rank of target doc
        for i, (doc, score) in enumerate(scored_docs):
            if "183 millones" in doc.page_content:
                print(f"\n>>> TARGET DOC RANK: {i+1} (Score: {score:.4f}) <<<")
                print(f"Content: {doc.page_content}")


    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    debug_retrieval()
