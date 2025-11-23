import os
import logging
from typing import List, Any
from core.interfaces.file_storage import FileStorageRepository
from infrastructure.constants import DIR_RAW_FILES

logger = logging.getLogger(__name__)

class LocalFileStorage(FileStorageRepository):
    """
    Implementación de almacenamiento en sistema de archivos local.
    """
    
    def _get_raw_files_dir(self, session_path: str) -> str:
        """Helper para obtener el directorio de archivos crudos."""
        path = os.path.join(session_path, DIR_RAW_FILES)
        os.makedirs(path, exist_ok=True)
        return path

    def save_file(self, session_path: str, filename: str, file_content: Any) -> str:
        try:
            raw_files_dir = self._get_raw_files_dir(session_path)
            file_path = os.path.join(raw_files_dir, filename)
            
            with open(file_path, "wb") as f:
                if hasattr(file_content, 'getbuffer'):
                    f.write(file_content.getbuffer())
                else:
                    f.write(file_content.read())
            
            logger.info(f"Archivo guardado exitosamente: {file_path}")
            return file_path
        except Exception as e:
            logger.error(f"Error guardando archivo {filename} en {session_path}: {e}")
            raise e

    def delete_file(self, session_path: str, filename: str) -> bool:
        raw_files_dir = self._get_raw_files_dir(session_path)
        file_path = os.path.join(raw_files_dir, filename)
        
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
                logger.info(f"Archivo eliminado: {file_path}")
                return True
            except OSError as e:
                logger.error(f"Error eliminando archivo {file_path}: {e}")
                return False
        return False # Retorna False si no existe, pero el servicio lo manejará

    def list_files(self, session_path: str) -> List[str]:
        try:
            raw_files_dir = self._get_raw_files_dir(session_path)
            if not os.path.exists(raw_files_dir):
                return []
            return [f for f in os.listdir(raw_files_dir) 
                    if os.path.isfile(os.path.join(raw_files_dir, f)) and not f.startswith('.')]
        except Exception as e:
            logger.error(f"Error listando archivos en {session_path}: {e}")
            return []

    def get_file_path(self, session_path: str, filename: str) -> str:
        raw_files_dir = self._get_raw_files_dir(session_path)
        return os.path.join(raw_files_dir, filename)

    def file_exists(self, session_path: str, filename: str) -> bool:
        return os.path.exists(self.get_file_path(session_path, filename))
