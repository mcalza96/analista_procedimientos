import json
import os
import logging
from pathlib import Path
from typing import List, Any
from infrastructure.constants import DIR_CHATS

logger = logging.getLogger(__name__)

class ChatIOHandler:
    """Manejador de operaciones de entrada/salida para archivos de chat."""

    @staticmethod
    def save_history(session_path: Path, chat_id: str, history: List[Any]) -> None:
        """Guarda el historial de un chat en disco."""
        try:
            chats_dir = session_path / DIR_CHATS
            chats_dir.mkdir(exist_ok=True)
            
            chat_path = chats_dir / f"{chat_id}.json"
            with open(chat_path, "w", encoding="utf-8") as f:
                json.dump(history, f, indent=4, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Error guardando historial de chat {chat_id}: {e}")

    @staticmethod
    def load_history(session_path: Path, chat_id: str) -> List[Any]:
        """Carga el historial de un chat desde disco."""
        chat_path = session_path / DIR_CHATS / f"{chat_id}.json"
        if not chat_path.exists():
            return []
            
        try:
            with open(chat_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            logger.error(f"Error decodificando historial de chat {chat_id}: {e}")
            return []
        except Exception as e:
            logger.error(f"Error leyendo historial de chat {chat_id}: {e}")
            return []

    @staticmethod
    def delete_history(session_path: Path, chat_id: str) -> bool:
        """Elimina el archivo de historial de un chat."""
        try:
            chat_file = session_path / DIR_CHATS / f"{chat_id}.json"
            if chat_file.exists():
                os.remove(chat_file)
                return True
            return False
        except Exception as e:
            logger.error(f"Error eliminando historial de chat {chat_id}: {e}")
            return False
