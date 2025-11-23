from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional

class SessionRepository(ABC):
    """Interfaz para el repositorio de sesiones."""

    @abstractmethod
    def create_session(self, name: str) -> str:
        """Crea una nueva sesión y retorna su ID."""
        pass

    @abstractmethod
    def delete_session(self, session_id: str) -> bool:
        """Elimina una sesión por su ID."""
        pass

    @abstractmethod
    def rename_session(self, session_id: str, new_name: str) -> bool:
        """Renombra una sesión existente."""
        pass

    @abstractmethod
    def get_session_name(self, session_id: str) -> str:
        """Obtiene el nombre de una sesión."""
        pass

    @abstractmethod
    def get_session_date(self, session_id: str) -> str:
        """Obtiene la fecha de creación de una sesión."""
        pass

    @abstractmethod
    def get_session_path(self, session_id: str) -> str:
        """Obtiene la ruta física de la sesión (si aplica)."""
        pass

    @abstractmethod
    def remove_file_from_session(self, session_id: str, filename: str):
        """Elimina la referencia de un archivo en la sesión."""
        pass

    @abstractmethod
    def update_session_summary(self, session_id: str, summary: str):
        """Actualiza el resumen de la sesión."""
        pass

    @abstractmethod
    def get_session_summary(self, session_id: str) -> Optional[str]:
        """Obtiene el resumen de la sesión."""
        pass

    # Chat Management Methods
    @abstractmethod
    def create_chat(self, session_id: str, title: str = None) -> str:
        """Crea un nuevo chat en la sesión."""
        pass

    @abstractmethod
    def list_chats(self, session_id: str) -> List[Dict[str, Any]]:
        """Lista los chats de una sesión."""
        pass

    @abstractmethod
    def delete_chat(self, session_id: str, chat_id: str) -> bool:
        """Elimina un chat específico."""
        pass

    @abstractmethod
    def save_chat_history(self, session_id: str, chat_id: str, history: List[Any]):
        """Guarda el historial de un chat."""
        pass

    @abstractmethod
    def load_chat_history(self, session_id: str, chat_id: str) -> List[Any]:
        """Carga el historial de un chat."""
        pass

    @abstractmethod
    def list_sessions(self) -> List[Dict[str, Any]]:
        """Lista todas las sesiones disponibles."""
        pass

    @abstractmethod
    def add_files_to_session(self, session_id: str, filenames: List[str]):
        """Añade archivos a los metadatos de la sesión."""
        pass

    @abstractmethod
    def get_session_files(self, session_id: str) -> List[str]:
        """Obtiene la lista de archivos de la sesión."""
        pass
