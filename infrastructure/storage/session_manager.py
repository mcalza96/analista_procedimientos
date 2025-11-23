import os
import uuid
import shutil
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
from core.interfaces.session_repository import SessionRepository
from infrastructure.storage.handlers.metadata_handler import MetadataHandler
from infrastructure.storage.handlers.chat_io_handler import ChatIOHandler
from infrastructure.constants import (
    DIR_SESSIONS, DIR_VECTOR_STORE, DIR_DOC_STORE, DIR_CHATS,
    FILE_METADATA, FILE_HISTORY_LEGACY, DEFAULT_SESSION_NAME, DEFAULT_CHAT_TITLE_PREFIX
)

class FileSessionRepository(SessionRepository):
    def __init__(self, base_path: str = f"data/{DIR_SESSIONS}"):
        """
        Inicializa el FileSessionRepository.
        
        Args:
            base_path (str): Ruta base donde se almacenarán las sesiones.
        """
        self.base_path = Path(base_path)
        self._ensure_base_path()

    def _ensure_base_path(self):
        """Asegura que el directorio base exista."""
        if not self.base_path.exists():
            self.base_path.mkdir(parents=True, exist_ok=True)

    def _get_validated_session_path(self, session_id: str) -> Path:
        """Obtiene y valida la ruta de la sesión."""
        session_path = self.base_path / session_id
        if not session_path.exists():
            raise ValueError(f"Session {session_id} does not exist")
        return session_path

    def create_session(self, name: str) -> str:
        """
        Crea una nueva sesión aislada.
        """
        session_id = str(uuid.uuid4())
        session_path = self.base_path / session_id
        
        # Crear estructura de directorios
        session_path.mkdir(parents=True, exist_ok=True)
        (session_path / DIR_VECTOR_STORE).mkdir(exist_ok=True)
        (session_path / DIR_DOC_STORE).mkdir(exist_ok=True)
        (session_path / DIR_CHATS).mkdir(exist_ok=True)
        
        # Crear metadatos
        metadata = {
            "id": session_id,
            "name": name,
            "created_at": datetime.now().isoformat(),
            "files": [],
            "chats": []
        }
        
        MetadataHandler.save(session_path, metadata)
            
        return session_id

    def create_chat(self, session_id: str, title: str = None) -> str:
        """
        Crea un nuevo chat dentro de una sesión.
        """
        session_path = self._get_validated_session_path(session_id)
            
        chat_id = str(uuid.uuid4())
        created_at = datetime.now().isoformat()
        
        if not title:
            title = f"{DEFAULT_CHAT_TITLE_PREFIX} {datetime.now().strftime('%d/%m %H:%M')}"
            
        metadata = MetadataHandler.load(session_path)
        if metadata:
            chats = metadata.get("chats", [])
            chats.append({
                "id": chat_id,
                "title": title,
                "created_at": created_at
            })
            metadata["chats"] = chats
            MetadataHandler.save(session_path, metadata)
            
        # Crear archivo de chat vacío
        ChatIOHandler.save_history(session_path, chat_id, [])
        
        return chat_id

    def list_chats(self, session_id: str) -> List[Dict[str, Any]]:
        """
        Lista los chats de una sesión.
        """
        try:
            session_path = self._get_validated_session_path(session_id)
        except ValueError:
            return []
            
        metadata = MetadataHandler.load(session_path)
        if not metadata:
            return []
            
        chats = metadata.get("chats", [])
        
        # Migración Legacy
        legacy_history = session_path / FILE_HISTORY_LEGACY
        if not chats and legacy_history.exists():
            legacy_chat_id = self.create_chat(session_id, "Historial Migrado")
            try:
                with open(legacy_history, "r", encoding="utf-8") as f:
                    history = json.load(f)
                ChatIOHandler.save_history(session_path, legacy_chat_id, history)
                os.remove(legacy_history)
                metadata = MetadataHandler.load(session_path)
                chats = metadata.get("chats", [])
            except Exception:
                pass
                
        chats.sort(key=lambda x: x.get("created_at", ""), reverse=True)
        return chats

    def delete_chat(self, session_id: str, chat_id: str) -> bool:
        """Elimina un chat específico."""
        try:
            session_path = self._get_validated_session_path(session_id)
        except ValueError:
            return False
            
        metadata = MetadataHandler.load(session_path)
        if metadata:
            chats = metadata.get("chats", [])
            new_chats = [c for c in chats if c["id"] != chat_id]
            
            if len(new_chats) < len(chats):
                metadata["chats"] = new_chats
                MetadataHandler.save(session_path, metadata)
                ChatIOHandler.delete_history(session_path, chat_id)
                return True
        return False

    def save_chat_history(self, session_id: str, chat_id: str, history: List[Any]):
        """Guarda el historial de un chat específico."""
        session_path = self._get_validated_session_path(session_id)
        ChatIOHandler.save_history(session_path, chat_id, history)

    def load_chat_history(self, session_id: str, chat_id: str) -> List[Any]:
        """Carga el historial de un chat específico."""
        try:
            session_path = self._get_validated_session_path(session_id)
        except ValueError:
            return []
        return ChatIOHandler.load_history(session_path, chat_id)

    def get_session_path(self, session_id: str) -> str:
        """Obtiene la ruta absoluta del directorio de la sesión."""
        session_path = self.base_path / session_id
        return str(session_path.resolve())

    def delete_session(self, session_id: str) -> bool:
        """Elimina una sesión y todos sus datos asociados."""
        try:
            session_path = self._get_validated_session_path(session_id)
        except ValueError:
            return False
        try:
            shutil.rmtree(session_path)
            return True
        except Exception:
            return False

    def rename_session(self, session_id: str, new_name: str) -> bool:
        """Renombra una sesión existente."""
        try:
            session_path = self._get_validated_session_path(session_id)
        except ValueError:
            return False
        return MetadataHandler.update(session_path, {"name": new_name})

    def remove_file_from_session(self, session_id: str, filename: str):
        """Elimina un archivo de los metadatos de la sesión."""
        try:
            session_path = self._get_validated_session_path(session_id)
        except ValueError:
            return
            
        metadata = MetadataHandler.load(session_path)
        if metadata:
            current_files = metadata.get("files", [])
            if filename in current_files:
                current_files.remove(filename)
                metadata["files"] = current_files
                MetadataHandler.save(session_path, metadata)

    def get_session_name(self, session_id: str) -> str:
        """Obtiene el nombre de la sesión."""
        try:
            session_path = self._get_validated_session_path(session_id)
        except ValueError:
            return DEFAULT_SESSION_NAME
        metadata = MetadataHandler.load(session_path)
        return metadata.get("name", DEFAULT_SESSION_NAME) if metadata else DEFAULT_SESSION_NAME

    def get_session_date(self, session_id: str) -> str:
        """Obtiene la fecha de creación de la sesión."""
        try:
            session_path = self._get_validated_session_path(session_id)
        except ValueError:
            return "Desconocida"
        metadata = MetadataHandler.load(session_path)
        if metadata and "created_at" in metadata:
            try:
                dt = datetime.fromisoformat(metadata["created_at"])
                return dt.strftime("%Y-%m-%d %H:%M")
            except ValueError:
                return metadata["created_at"]
        return "Desconocida"

    def update_session_summary(self, session_id: str, summary: str):
        """Actualiza el resumen ejecutivo de la sesión."""
        try:
            session_path = self._get_validated_session_path(session_id)
        except ValueError:
            return
        MetadataHandler.update(session_path, {"summary": summary})

    def get_session_summary(self, session_id: str) -> Optional[str]:
        """Obtiene el resumen ejecutivo de la sesión."""
        try:
            session_path = self._get_validated_session_path(session_id)
        except ValueError:
            return None
        metadata = MetadataHandler.load(session_path)
        return metadata.get("summary") if metadata else None

    def list_sessions(self) -> List[Dict[str, Any]]:
        """Lista todas las sesiones disponibles."""
        sessions = []
        if not self.base_path.exists():
            return sessions
            
        for item in self.base_path.iterdir():
            if item.is_dir():
                metadata = MetadataHandler.load(item)
                if metadata:
                    sessions.append(metadata)
        
        sessions.sort(key=lambda x: x.get("created_at", ""), reverse=True)
        return sessions

    def add_files_to_session(self, session_id: str, filenames: List[str]):
        """Añade archivos a los metadatos de la sesión."""
        try:
            session_path = self._get_validated_session_path(session_id)
        except ValueError:
            return
            
        metadata = MetadataHandler.load(session_path)
        if metadata:
            current_files = metadata.get("files", [])
            for f in filenames:
                if f not in current_files:
                    current_files.append(f)
            
            metadata["files"] = current_files
            MetadataHandler.save(session_path, metadata)

    def get_session_files(self, session_id: str) -> List[str]:
        """Obtiene la lista de archivos de la sesión."""
        try:
            session_path = self._get_validated_session_path(session_id)
        except ValueError:
            return []
        metadata = MetadataHandler.load(session_path)
        return metadata.get("files", []) if metadata else []
