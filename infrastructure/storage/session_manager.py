import os
import json
import uuid
import shutil
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

class SessionManager:
    def __init__(self, base_path: str = "data/sessions"):
        """
        Inicializa el SessionManager.
        
        Args:
            base_path (str): Ruta base donde se almacenarán las sesiones.
        """
        self.base_path = Path(base_path)
        self._ensure_base_path()

    def _ensure_base_path(self):
        """Asegura que el directorio base exista."""
        if not self.base_path.exists():
            self.base_path.mkdir(parents=True, exist_ok=True)

    def create_session(self, name: str) -> str:
        """
        Crea una nueva sesión aislada.
        
        Args:
            name (str): Nombre legible para la sesión.
            
        Returns:
            str: El ID único de la sesión creada.
        """
        session_id = str(uuid.uuid4())
        session_path = self.base_path / session_id
        
        # Crear estructura de directorios
        session_path.mkdir(parents=True, exist_ok=True)
        (session_path / "vector_store").mkdir(exist_ok=True)
        (session_path / "doc_store").mkdir(exist_ok=True)
        
        # Crear metadatos
        metadata = {
            "id": session_id,
            "name": name,
            "created_at": datetime.now().isoformat(),
            "files": []
        }
        
        self._save_metadata(session_path, metadata)
            
        return session_id

    def add_files_to_session(self, session_id: str, filenames: List[str]):
        """
        Registra archivos agregados a la sesión.
        
        Args:
            session_id (str): ID de la sesión.
            filenames (List[str]): Lista de nombres de archivos.
        """
        session_path = self.base_path / session_id
        if not session_path.exists():
            raise ValueError(f"Session {session_id} does not exist")
            
        metadata = self._load_metadata(session_path)
        if metadata:
            current_files = metadata.get("files", [])
            # Evitar duplicados
            for f in filenames:
                if f not in current_files:
                    current_files.append(f)
            metadata["files"] = current_files
            self._save_metadata(session_path, metadata)

    def get_session_files(self, session_id: str) -> List[str]:
        """
        Obtiene la lista de archivos registrados en la sesión.
        
        Args:
            session_id (str): ID de la sesión.
            
        Returns:
            List[str]: Lista de nombres de archivos.
        """
        session_path = self.base_path / session_id
        if not session_path.exists():
            return []
            
        metadata = self._load_metadata(session_path)
        return metadata.get("files", []) if metadata else []

    def list_sessions(self) -> List[Dict[str, Any]]:
        """
        Lista todas las sesiones existentes ordenadas por fecha (más reciente primero).
        
        Returns:
            List[Dict[str, Any]]: Lista de metadatos de las sesiones.
        """
        sessions = []
        if not self.base_path.exists():
            return sessions
            
        for session_dir in self.base_path.iterdir():
            if session_dir.is_dir():
                metadata = self._load_metadata(session_dir)
                if metadata:
                    sessions.append(metadata)
                        
        # Ordenar por fecha de creación descendente
        sessions.sort(key=lambda x: x.get("created_at", ""), reverse=True)
        return sessions

    def save_history(self, session_id: str, history: List[Any]):
        """
        Guarda el historial del chat en la sesión.
        
        Args:
            session_id (str): ID de la sesión.
            history (List[Any]): Lista de mensajes a guardar.
        """
        session_path = self.base_path / session_id
        if not session_path.exists():
            raise ValueError(f"Session {session_id} does not exist")
            
        history_path = session_path / "history.json"
        with open(history_path, "w", encoding="utf-8") as f:
            # Se asume que history es serializable a JSON (lista de dicts)
            # Si son objetos complejos, el llamador debería convertirlos antes
            json.dump(history, f, indent=4, ensure_ascii=False)

    def load_history(self, session_id: str) -> List[Any]:
        """
        Carga el historial del chat de la sesión.
        
        Args:
            session_id (str): ID de la sesión.
            
        Returns:
            List[Any]: Lista de mensajes recuperada.
        """
        session_path = self.base_path / session_id
        if not session_path.exists():
            raise ValueError(f"Session {session_id} does not exist")
            
        history_path = session_path / "history.json"
        if not history_path.exists():
            return []
            
        try:
            with open(history_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError:
            return []

    def get_session_path(self, session_id: str) -> str:
        """
        Obtiene la ruta absoluta del directorio de la sesión.
        
        Args:
            session_id (str): ID de la sesión.
            
        Returns:
            str: Ruta absoluta.
        """
        session_path = self.base_path / session_id
        return str(session_path.resolve())

    def delete_session(self, session_id: str) -> bool:
        """
        Elimina una sesión y todos sus datos asociados.
        
        Args:
            session_id (str): ID de la sesión a eliminar.
            
        Returns:
            bool: True si se eliminó correctamente, False si no existía o hubo error.
        """
        session_path = self.base_path / session_id
        if not session_path.exists():
            return False
            
        try:
            shutil.rmtree(session_path)
            return True
        except Exception:
            return False

    def _save_metadata(self, session_path: Path, metadata: Dict[str, Any]):
        """Helper para guardar metadatos."""
        with open(session_path / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=4, ensure_ascii=False)

    def _load_metadata(self, session_path: Path) -> Optional[Dict[str, Any]]:
        """Helper para cargar metadatos."""
        metadata_path = session_path / "metadata.json"
        if not metadata_path.exists():
            return None
        try:
            with open(metadata_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None

    def get_session_name(self, session_id: str) -> str:
        """
        Obtiene el nombre de la sesión.
        
        Args:
            session_id (str): ID de la sesión.
            
        Returns:
            str: Nombre de la sesión o "Sesión Desconocida".
        """
        session_path = self.base_path / session_id
        metadata = self._load_metadata(session_path)
        return metadata.get("name", "Sesión Desconocida") if metadata else "Sesión Desconocida"

    def get_session_date(self, session_id: str) -> str:
        """
        Obtiene la fecha de creación de la sesión.
        
        Args:
            session_id (str): ID de la sesión.
            
        Returns:
            str: Fecha formateada o "Desconocida".
        """
        session_path = self.base_path / session_id
        metadata = self._load_metadata(session_path)
        if metadata and "created_at" in metadata:
            try:
                dt = datetime.fromisoformat(metadata["created_at"])
                return dt.strftime("%Y-%m-%d %H:%M")
            except ValueError:
                return metadata["created_at"]
        return "Desconocida"
