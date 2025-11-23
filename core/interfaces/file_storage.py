from abc import ABC, abstractmethod
from typing import List, Any, BinaryIO

class FileStorageRepository(ABC):
    """
    Interfaz abstracta para el almacenamiento de archivos.
    Permite desacoplar la lógica de negocio del sistema de archivos local.
    """
    
    @abstractmethod
    def save_file(self, session_path: str, filename: str, file_content: Any) -> str:
        """Guarda un archivo y retorna su ruta absoluta o identificador."""
        pass

    @abstractmethod
    def delete_file(self, session_path: str, filename: str) -> bool:
        """Elimina un archivo."""
        pass

    @abstractmethod
    def list_files(self, session_path: str) -> List[str]:
        """Lista los nombres de archivos en una sesión."""
        pass
    
    @abstractmethod
    def get_file_path(self, session_path: str, filename: str) -> str:
        """Obtiene la ruta completa de un archivo."""
        pass
    
    @abstractmethod
    def file_exists(self, session_path: str, filename: str) -> bool:
        """Verifica si un archivo existe."""
        pass
