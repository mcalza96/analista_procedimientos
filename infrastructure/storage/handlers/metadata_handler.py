import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from infrastructure.constants import FILE_METADATA

logger = logging.getLogger(__name__)

class MetadataHandler:
    """Manejador de operaciones de lectura/escritura de metadatos de sesión."""
    
    @staticmethod
    def load(session_path: Path) -> Optional[Dict[str, Any]]:
        """Carga los metadatos de una sesión."""
        metadata_path = session_path / FILE_METADATA
        if not metadata_path.exists():
            return None
        try:
            with open(metadata_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            logger.error(f"Error decodificando metadatos en {metadata_path}: {e}")
            return None
        except Exception as e:
            logger.error(f"Error leyendo metadatos en {metadata_path}: {e}")
            return None

    @staticmethod
    def save(session_path: Path, metadata: Dict[str, Any]) -> None:
        """Guarda los metadatos de una sesión."""
        try:
            with open(session_path / FILE_METADATA, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=4, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Error guardando metadatos en {session_path}: {e}")

    @staticmethod
    def update(session_path: Path, updates: Dict[str, Any]) -> bool:
        """Actualiza campos específicos de los metadatos."""
        metadata = MetadataHandler.load(session_path)
        if metadata:
            metadata.update(updates)
            MetadataHandler.save(session_path, metadata)
            return True
        return False
