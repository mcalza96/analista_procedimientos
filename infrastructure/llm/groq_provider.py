import logging
import json
import random
from typing import List, Any
from langchain_groq import ChatGroq
from langchain_classic.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain_classic.retrievers import EnsembleRetriever
from core.interfaces.llm_provider import LLMProvider
from core.domain.models import ChatResponse, SourceDocument, Quiz, QuizQuestion
from config.settings import settings
from core.services.router import SemanticRouter # We will move router later or keep it in src for now

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
        self.router = SemanticRouter()

    def generate_response(self, query: str, context: List[Any], chat_history: List[Any], route: str = None) -> ChatResponse:
        try:
            # 1. Routing
            if not route:
                try:
                    route = self.router.route(query)
                except Exception as e:
                    logger.error(f"Routing error: {e}")
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
            # Context here is expected to be (vectorstore, bm25_retriever)
            vectorstore, bm25_retriever = context
            
            if not vectorstore or not bm25_retriever:
                 return ChatResponse(answer="Por favor, carga documentos primero.", route="ERROR")

            faiss_retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 4})
            ensemble_retriever = EnsembleRetriever(
                retrievers=[bm25_retriever, faiss_retriever],
                weights=[0.5, 0.5]
            )

            # Select Prompt
            if route == "ANALYSIS":
                sys_prompt = """Eres un Auditor de Calidad ISO 9001 estricto.
Extrae evidencia concreta. Si no está explícito, di 'No hay información suficiente'.
Cita el documento y sección usando referencias numéricas [1].
Contexto: {context}"""
            elif route == "WALKTHROUGH":
                sys_prompt = """Eres un Instructor de Laboratorio.
Guía paso a paso. Si dice 'Empezar', da el Paso 1.
Contexto: {context}"""
            else: # PRECISION
                sys_prompt = """Eres un Asistente Técnico estricto.
Responde basándote ÚNICAMENTE en el contexto. Sé breve.
Cita usando referencias numéricas [1].
Contexto: {context}"""

            # Chains
            contextualize_q_prompt = ChatPromptTemplate.from_messages([
                ("system", "Reformulate the question to be standalone given the chat history."),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ])
            
            history_aware_retriever = create_history_aware_retriever(
                self.llm_rag, ensemble_retriever, contextualize_q_prompt
            )
            
            qa_prompt = ChatPromptTemplate.from_messages([
                ("system", sys_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ])
            
            document_prompt = PromptTemplate(
                input_variables=["page_content", "source"],
                template="Content: {page_content}\nSource: {source}"
            )
            
            question_answer_chain = create_stuff_documents_chain(
                self.llm_rag, qa_prompt, document_prompt=document_prompt
            )
            
            rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
            
            response = rag_chain.invoke({"input": query, "chat_history": chat_history})
            
            # Process Sources
            source_docs = []
            for doc in response.get("context", []):
                source_docs.append(SourceDocument(
                    page_content=doc.page_content,
                    metadata=doc.metadata,
                    source_file=doc.metadata.get('source_file', ''),
                    page_number=doc.metadata.get('page', 0)
                ))

            return ChatResponse(
                answer=response["answer"],
                source_documents=source_docs,
                route=route
            )

        except Exception as e:
            logger.error(f"Error in generate_response: {e}")
            return ChatResponse(answer="Error procesando la solicitud.", route="ERROR")

    def generate_quiz(self, topic: str, difficulty: str, num_questions: int, context: List[Any]) -> Quiz:
        # Context is (vectorstore, bm25_retriever)
        vectorstore, bm25_retriever = context
        
        # Retrieve relevant content for the topic
        docs = []
        if vectorstore:
            # 1. Retrieve a larger pool of documents to ensure variety
            docs = vectorstore.similarity_search(topic, k=20)
        
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
