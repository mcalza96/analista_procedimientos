import os
import fitz
from PIL import Image
from io import BytesIO
import streamlit as st
from streamlit_pdf_viewer import pdf_viewer
from config.settings import settings

def render_page_image(filename, page_num):
    try:
        full_path = os.path.join(settings.TEMP_DOCS_DIR, filename)
        if not os.path.exists(full_path):
            return None
        
        pdf_document = fitz.open(full_path)
        if page_num < 0 or page_num >= len(pdf_document):
            pdf_document.close()
            return None
        
        page = pdf_document[page_num]
        mat = fitz.Matrix(2, 2)
        pix = page.get_pixmap(matrix=mat)
        img_bytes = pix.tobytes("png")
        pdf_document.close()
        img = Image.open(BytesIO(img_bytes))
        return img
    except Exception:
        return None

def display_sources(sources, text_content=""):
    if not sources:
        return

    relevant_indices = set()
    if text_content:
        import re
        matches = re.findall(r'\[(\d+)\]', text_content)
        for match in matches:
            try:
                idx = int(match) - 1
                if 0 <= idx < len(sources):
                    relevant_indices.add(idx)
            except ValueError:
                pass

    if not relevant_indices and text_content:
        for i, doc in enumerate(sources):
            source_name = os.path.basename(doc.metadata.get('source', ''))
            if source_name in text_content:
                relevant_indices.add(i)
    
    if relevant_indices:
        final_sources = [sources[i] for i in sorted(list(relevant_indices))]
    else:
        final_sources = sources

    if not final_sources:
        return

    st.markdown("### 📚 Fuentes Utilizadas")
    
    if len(final_sources) <= 3:
        cols = st.columns(len(final_sources))
        for i, (col, doc) in enumerate(zip(cols, final_sources)):
            with col:
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
                    
                    if source_file:
                        img = render_page_image(source_file, page - 1) # page is 1-indexed in metadata usually? No, usually 0-indexed in loader but displayed as +1. Let's check loader.
                        # Loader uses PyPDFLoader. metadata['page'] is 0-indexed.
                        # In display_sources original code: page = doc.metadata.get('page', 0) + 1
                        # render_page_image expects 0-indexed.
                        if img:
                            st.image(img)
                        else:
                            st.caption("_Sin vista previa_")
                    
                    popover_key = f"popover_{original_idx}_{i}"
                    with st.popover("🔍 Ver PDF Completo", use_container_width=True):
                        st.markdown(f"**Visualizando: {source} (Pág. {page})**")
                        if source_file:
                            full_path = os.path.join(settings.TEMP_DOCS_DIR, source_file)
                            if os.path.exists(full_path):
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
        with st.expander(f"Ver {len(final_sources)} documentos consultados"):
            for i, doc in enumerate(final_sources):
                try:
                    original_idx = sources.index(doc) + 1
                except ValueError:
                    original_idx = i + 1

                source = os.path.basename(doc.metadata.get('source', 'Desconocido'))
                page = doc.metadata.get('page', 0) + 1
                source_file = doc.metadata.get('source_file', '')
                
                st.markdown(f"**[{original_idx}] {source}** (Pág. {page})")
                
                if st.button(f"🔍 Ver PDF [{original_idx}]", key=f"btn_pdf_{i}"):
                     if source_file:
                        full_path = os.path.join(settings.TEMP_DOCS_DIR, source_file)
                        if os.path.exists(full_path):
                            pdf_viewer(full_path, width=700, height=600, pages_to_render=[page])
                
                st.caption(f"_{doc.page_content[:200]}..._")
                st.divider()
