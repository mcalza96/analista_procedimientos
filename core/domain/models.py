from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

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
    route: str = "DEFAULT"

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
