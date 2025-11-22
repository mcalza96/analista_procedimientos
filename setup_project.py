import os
from pathlib import Path

def create_structure():
    # Definición de la estructura del proyecto
    directories = [
        "src",
        "docs_temp"
    ]
    
    files = [
        ".env",
        ".gitignore",
        "requirements.txt",
        "main.py",
        "src/__init__.py",
        "src/document_loader.py",
        "src/vector_store.py",
        "src/llm_engine.py"
    ]

    base_path = Path.cwd()
    print(f"Iniciando configuración del proyecto en: {base_path}")

    # Crear directorios
    for directory in directories:
        dir_path = base_path / directory
        try:
            dir_path.mkdir(parents=True, exist_ok=True)
            print(f"✅ Directorio creado: {directory}")
        except Exception as e:
            print(f"❌ Error creando directorio {directory}: {e}")

    # Crear archivos vacíos
    for file in files:
        file_path = base_path / file
        try:
            if not file_path.exists():
                file_path.touch()
                print(f"✅ Archivo creado: {file}")
            else:
                print(f"ℹ️  El archivo ya existe: {file}")
        except Exception as e:
            print(f"❌ Error creando archivo {file}: {e}")

if __name__ == "__main__":
    create_structure()
