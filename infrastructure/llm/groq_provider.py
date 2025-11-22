import logging
import json
import random
from typing import List, Any
from langchain_groq import ChatGroq
from langchain_classic.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain_classic.retrievers import EnsembleRetriever
from sentence_transformers import CrossEncoder
from core.interfaces.llm_provider import LLMProvider
from core.domain.models import ChatResponse, SourceDocument, Quiz, QuizQuestion
from config.settings import settings

logger = logging.getLogger(__name__)

class GroqProvider(LLMProvider):
    def __init__(self):
        if not settings.GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY not found")
        
        self.llm_chat = ChatGroq(
            groq_api_key=settings.GROQ_API_KEY,
            model_name=settings.MODEL_NAME,
            temperature=0.7
        )
        self.llm_rag = ChatGroq(
            groq_api_key=settings.GROQ_API_KEY,
            model_name=settings.MODEL_NAME,
            temperature=0.1
        )
        self.reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

    def generate_response(self, query: str, context: List[Any], chat_history: List[Any], route: str = None) -> ChatResponse:
        try:
            # 1. Routing
            if not route:
                route = "PRECISION"
            
            # 2. Chat Mode
            if route == "CHAT":
                prompt = ChatPromptTemplate.from_template("""Eres un asistente amable del laboratorio ISO 9001.
Responde al saludo o pregunta general de forma cordial y profesional.
Si te preguntan sobre procedimientos, sugiere subir manuales.
Pregunta: {query}""")
                chain = prompt | self.llm_chat
                res = chain.invoke({"query": query})
                return ChatResponse(answer=res.content, route=route)

            # 3. RAG Mode
            # Context here is expected to be (parent_retriever, bm25_retriever)
            parent_retriever, bm25_retriever = context
            
            if not parent_retriever or not bm25_retriever:
                 return ChatResponse(answer="Por favor, carga documentos primero.", route="ERROR")

            # Ensemble Retriever
            # Combinamos el ParentDocumentRetriever (semántico + jerarquía) con BM25 (keywords)
            # Damos más peso al Parent Retriever porque es más robusto semánticamente
            ensemble_retriever = EnsembleRetriever(
                retrievers=[bm25_retriever, parent_retriever],
                weights=[0.4, 0.6]
            )

            # Obtén los documentos iniciales
            initial_docs = ensemble_retriever.invoke(query)

            # Reranking
            pairs = [[query, doc.page_content] for doc in initial_docs]
            scores = self.reranker.predict(pairs)
            
            # Zip docs with scores and sort
            scored_docs = sorted(zip(initial_docs, scores), key=lambda x: x[1], reverse=True)
            
            # Select Top 5
            top_docs = [doc for doc, score in scored_docs[:5]]

            # Select Prompt
            if route == "ANALYSIS":
                sys_prompt = """Eres un Auditor de Calidad ISO 9001 estricto y profesional.
Tu objetivo es analizar el contexto proporcionado y extraer evidencia concreta para responder a la consulta.

REGLAS DE CITA ESTRICTAS:
1. CADA afirmación o dato extraído del texto DEBE llevar una cita numérica [X] INMEDIATAMENTE al final de la frase correspondiente.
2. NO coloques las citas al final del párrafo, deben ser precisas por frase.
3. Si una frase se basa en múltiples fuentes, usa el formato [1][2].
4. NO generes una sección de 'Fuentes', 'Referencias' o 'Bibliografía' al final.
5. Si la información no está explícita en el contexto, indica claramente 'No hay información suficiente en los documentos proporcionados'.

Contexto: {context}"""
            elif route == "WALKTHROUGH":
                sys_prompt = """Eres un Instructor de Laboratorio.
Guía paso a paso. Si dice 'Empezar', da el Paso 1.
Contexto: {context}"""
            else: # PRECISION
                sys_prompt = """Eres un Asistente Técnico estricto y directo.
Responde basándote ÚNICAMENTE en el contexto proporcionado.

REGLAS DE CITA ESTRICTAS:
1. Sé extremadamente breve y conciso.
2. CADA afirmación DEBE llevar su cita numérica [X] INMEDIATAMENTE al final de la frase.
3. Si hay múltiples fuentes, usa [1][2].
4. NO generes listas de fuentes o bibliografía al final.
5. Si no encuentras la respuesta exacta, di 'No se encuentra en el contexto'.

Contexto: {context}"""

            # Chains
            qa_prompt = ChatPromptTemplate.from_messages([
                ("system", sys_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ])
            
            document_prompt = PromptTemplate(
                input_variables=["page_content", "source_file"],
                template="Content: {page_content}\nSource: {source_file}"
            )
            
            question_answer_chain = create_stuff_documents_chain(
                self.llm_rag, qa_prompt, document_prompt=document_prompt
            )
            
            response_message = question_answer_chain.invoke({
                "input": query, 
                "chat_history": chat_history,
                "context": top_docs
            })
            
            # Process Sources
            source_docs = []
            for doc in top_docs:
                source_docs.append(SourceDocument(
                    page_content=doc.page_content,
                    metadata=doc.metadata,
                    source_file=doc.metadata.get('source_file', ''),
                    page_number=doc.metadata.get('page', 0)
                ))

            return ChatResponse(
                answer=response_message.content if hasattr(response_message, 'content') else str(response_message),
                source_documents=source_docs,
                route=route
            )

        except Exception as e:
            logger.error(f"Error in generate_response: {e}")
            return ChatResponse(answer="Error procesando la solicitud.", route="ERROR")

    def generate_quiz(self, topic: str, difficulty: str, num_questions: int, context: List[Any]) -> Quiz:
        # Context is (retriever_or_vectorstore, bm25_retriever)
        retriever_component, bm25_retriever = context
        
        # Retrieve relevant content for the topic
        docs = []
        if retriever_component:
            # 1. Retrieve a larger pool of documents to ensure variety
            if hasattr(retriever_component, 'similarity_search'):
                docs = retriever_component.similarity_search(topic, k=20)
            else:
                # It's a retriever (ParentDocumentRetriever)
                # invoke returns relevant documents (parents)
                docs = retriever_component.invoke(topic)
                # If we want more, we might need to adjust search_kwargs of the underlying vectorstore?
                # ParentDocumentRetriever uses search_kwargs of its vectorstore retriever if configured.
                # But invoke is standard.
        
        # 2. Randomly select a subset (e.g., 5) to force different questions each time
        if len(docs) > 5:
            docs = random.sample(docs, 5)
        
        context_chunks = []
        for d in docs:
            source = d.metadata.get('source_file', 'unknown')
            page = d.metadata.get('page', 0)
            context_chunks.append(f"Source: {source} | Page: {page}\nContent: {d.page_content}")
        context_text = "\n\n".join(context_chunks)
        
        prompt = f"""
        Eres un experto en formación ISO 9001. Genera un examen de opción múltiple.
        Tema: {topic}
        Dificultad: {difficulty}
        Cantidad: {num_questions} preguntas.
        
        Basado en este contexto:
        {context_text}
        
        INSTRUCCIONES CRÍTICAS:
        1. Genera {num_questions} preguntas basadas EXCLUSIVAMENTE en el contexto.
        2. Para cada pregunta, DEBES extraer el "Source" y "Page" exactos del fragmento de texto que usaste.
        3. Si usaste múltiples fragmentos, elige el más relevante.
        4. Devuelve SOLO un JSON válido.

        Formato JSON esperado:
        {{
            "topic": "{topic}",
            "questions": [
                {{
                    "question": "Texto de la pregunta",
                    "options": ["Opción A", "Opción B", "Opción C", "Opción D"],
                    "correct_answer": 0,
                    "explanation": "Por qué es correcta",
                    "source_file": "nombre_exacto_del_archivo.pdf", 
                    "page_number": 0 
                }}
            ]
        }}
        """
        
        response = self.llm_chat.invoke(prompt)
        content = response.content
        
        # Clean JSON
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0]
        elif "```" in content:
            content = content.split("```")[1].split("```")[0]
            
        try:
            data = json.loads(content)
            questions = []
            for q in data["questions"]:
                questions.append(QuizQuestion(
                    question=q["question"],
                    options=q["options"],
                    correct_answer=q["correct_answer"],
                    explanation=q.get("explanation", ""),
                    source_file=q.get("source_file"),
                    page_number=q.get("page_number")
                ))
            return Quiz(topic=data["topic"], questions=questions)
        except Exception as e:
            logger.error(f"Error parsing quiz JSON: {e}")
            return Quiz(topic=topic, questions=[])
