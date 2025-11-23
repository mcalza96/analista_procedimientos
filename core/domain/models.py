from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from enum import Enum

class RouteType(str, Enum):
    PRECISION = "PRECISION"
    ANALYSIS = "ANALYSIS"
    CHAT = "CHAT"
    WALKTHROUGH = "WALKTHROUGH"
    ERROR = "ERROR"

@dataclass
class SourceDocument:
    page_content: str
    metadata: Dict[str, Any]
    source_file: str = ""
    page_number: int = 0

@dataclass
class ChatResponse:
    answer: str
    source_documents: List[SourceDocument] = field(default_factory=list)
    route: RouteType = RouteType.CHAT

@dataclass
class QuizQuestion:
    question: str
    options: List[str]
    correct_answer: int  # Index of the correct option
    explanation: str
    source_file: Optional[str] = None
    page_number: Optional[int] = None

@dataclass
class Quiz:
    topic: str
    questions: List[QuizQuestion]

class LLMProviderError(Exception):
    """Excepción personalizada para errores del proveedor de LLM."""
    pass
