import os
import shutil
import csv
from datetime import datetime
from io import BytesIO
import fitz  # PyMuPDF
from PIL import Image
import streamlit as st
from streamlit_pdf_viewer import pdf_viewer

def save_uploaded_files(uploaded_files):
    """Guarda los archivos subidos en la carpeta temporal."""
    temp_dir = "docs_temp"
    saved_paths = []
    
    # Limpiar archivos antiguos ANTES de guardar nuevos
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    os.makedirs(temp_dir)
        
    for uploaded_file in uploaded_files:
        file_path = os.path.join(temp_dir, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        saved_paths.append(file_path)
        
    return saved_paths

def log_feedback(query, response, rating, details=""):
    """
    Registra el feedback del usuario en un archivo CSV para auditoría ISO 9001.
    
    Args:
        query (str): Pregunta del usuario.
        response (str): Respuesta del asistente.
        rating (str): "Positiva" o "Negativa".
        details (str): Detalles adicionales (especialmente para feedback negativo).
    """
    feedback_file = "feedback_log.csv"
    
    # Crear archivo con encabezados si no existe
    file_exists = os.path.isfile(feedback_file)
    
    with open(feedback_file, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        
        # Escribir encabezado solo si es un archivo nuevo
        if not file_exists:
            writer.writerow(["Timestamp", "Pregunta", "Respuesta", "Calificación", "Detalle"])
        
        # Escribir fila de feedback
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        writer.writerow([timestamp, query, response[:500], rating, details])  # Limitar respuesta a 500 chars

def render_page_image(filename, page_num):
    """
    Genera una imagen (miniatura) de una página específica de un PDF.
    
    Args:
        filename (str): Nombre del archivo PDF (sin ruta).
        page_num (int): Número de página (0-indexed).
        
    Returns:
        PIL.Image or None: Imagen de la página o None si falla.
    """
    try:
        # Construir la ruta completa al archivo en docs_temp/
        full_path = os.path.join("docs_temp", filename)
        
        # Verificar que el archivo existe
        if not os.path.exists(full_path):
            print(f"[DEBUG] Archivo no encontrado: {full_path}")
            return None
        
        # Abrir el documento PDF
        pdf_document = fitz.open(full_path)
        
        # Verificar que el número de página es válido
        if page_num < 0 or page_num >= len(pdf_document):
            pdf_document.close()
            print(f"[DEBUG] Número de página inválido: {page_num} (total: {len(pdf_document)})")
            return None
        
        # Cargar la página específica
        page = pdf_document[page_num]
        
        # Generar imagen con zoom para buena calidad
        # Matrix(2, 2) = zoom 2x para mejor resolución
        mat = fitz.Matrix(2, 2)
        pix = page.get_pixmap(matrix=mat)
        
        # Convertir pixmap a bytes
        img_bytes = pix.tobytes("png")
        
        # Cerrar documento
        pdf_document.close()
        
        # Convertir bytes a PIL Image
        img = Image.open(BytesIO(img_bytes))
        
        return img
        
    except Exception as e:
        # Log del error (opcional)
        print(f"[DEBUG] Error renderizando página {page_num} de {filename}: {e}")
        return None

def display_sources(sources, text_content=""):
    """
    Muestra las fuentes consultadas, filtrando solo las relevantes si es posible.
    Soporta citas numéricas tipo [1], [2] y coincidencia de nombres de archivo.
    Incluye visualización de PDF embebido.
    
    Args:
        sources (list): Lista de documentos recuperados.
        text_content (str): Texto de la respuesta generada (para filtrar citas).
    """
    if not sources:
        return

    relevant_indices = set()
    
    # 1. Detectar citas numéricas [1], [2], etc.
    if text_content:
        import re
        # Buscar patrones [1], [2], etc.
        matches = re.findall(r'\[(\d+)\]', text_content)
        for match in matches:
            try:
                idx = int(match) - 1  # Convertir a 0-indexed
                if 0 <= idx < len(sources):
                    relevant_indices.add(idx)
            except ValueError:
                pass

    # 2. Si no hay citas numéricas, intentar por nombre de archivo (Legacy)
    if not relevant_indices and text_content:
        for i, doc in enumerate(sources):
            source_name = os.path.basename(doc.metadata.get('source', ''))
            if source_name in text_content:
                relevant_indices.add(i)
    
    # 3. Selección final de fuentes
    if relevant_indices:
        final_sources = [sources[i] for i in sorted(list(relevant_indices))]
    else:
        # Fallback: Mostrar todas las fuentes recuperadas (contexto completo)
        final_sources = sources

    # Si después de todo no hay fuentes, salir
    if not final_sources:
        return

    # UI Mejorada: Usar un contenedor más limpio
    st.markdown("### 📚 Fuentes Utilizadas")
    
    # Usar columnas para mostrar fuentes como "tarjetas" si son pocas (<= 3)
    if len(final_sources) <= 3:
        cols = st.columns(len(final_sources))
        for i, (col, doc) in enumerate(zip(cols, final_sources)):
            with col:
                # Determinar el índice original para mostrar [N]
                try:
                    original_idx = sources.index(doc) + 1
                except ValueError:
                    original_idx = "?"
                
                source = os.path.basename(doc.metadata.get('source', 'Desconocido'))
                page = doc.metadata.get('page', 0) + 1
                source_file = doc.metadata.get('source_file', '')
                
                with st.container(border=True):
                    st.markdown(f"**[{original_idx}] 📄 {source}**")
                    st.caption(f"Página {page}")
                    
                    # Miniatura (Legacy)
                    page_num = doc.metadata.get('page', 0)
                    if source_file:
                        img = render_page_image(source_file, page_num)
                        if img:
                            st.image(img)
                        else:
                            st.caption("_Sin vista previa_")
                    
                    # Botón para ver PDF completo
                    # Usamos un key único para el popover
                    popover_key = f"popover_{original_idx}_{i}"
                    with st.popover("🔍 Ver PDF Completo", use_container_width=True):
                        st.markdown(f"**Visualizando: {source} (Pág. {page})**")
                        if source_file:
                            full_path = os.path.join("docs_temp", source_file)
                            if os.path.exists(full_path):
                                # Renderizar PDF Viewer
                                # pages_to_render espera una lista de números de página (1-indexed)
                                pdf_viewer(
                                    full_path, 
                                    width=700, 
                                    height=800, 
                                    pages_to_render=[page], 
                                    render_text=True
                                )
                            else:
                                st.error("Archivo no encontrado.")
                        
                    with st.expander("Ver texto extraído"):
                        st.caption(doc.page_content)
    else:
        # Si son muchas, usar el expander clásico pero más limpio
        with st.expander(f"Ver {len(final_sources)} documentos consultados"):
            for i, doc in enumerate(final_sources):
                # Determinar el índice original
                try:
                    original_idx = sources.index(doc) + 1
                except ValueError:
                    original_idx = i + 1

                source = os.path.basename(doc.metadata.get('source', 'Desconocido'))
                page = doc.metadata.get('page', 0) + 1
                source_file = doc.metadata.get('source_file', '')
                
                st.markdown(f"**[{original_idx}] {source}** (Pág. {page})")
                
                # Botón para ver PDF en expander
                if st.button(f"🔍 Ver PDF [{original_idx}]", key=f"btn_pdf_{i}"):
                     if source_file:
                        full_path = os.path.join("docs_temp", source_file)
                        if os.path.exists(full_path):
                            pdf_viewer(full_path, width=700, height=600, pages_to_render=[page])
                
                st.caption(f"_{doc.page_content[:200]}..._")
                st.divider()
