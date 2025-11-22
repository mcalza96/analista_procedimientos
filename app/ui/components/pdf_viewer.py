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

    # If no specific citations found, maybe show top 3?
    # But the requirement is "Clickable Citations".
    # If text_content is present but no citations, we might want to show the top sources anyway.
    
    if relevant_indices:
        final_sources = [sources[i] for i in sorted(list(relevant_indices))]
    else:
        # If no citations, show top 3 as fallback
        final_sources = sources[:3]

    if not final_sources:
        return

    st.markdown("### 📚 Fuentes Citadas")
    
    # Source Navigation Bar
    # We use columns to create a horizontal bar of buttons
    cols = st.columns(len(final_sources))
    
    for i, (col, doc) in enumerate(zip(cols, final_sources)):
        with col:
            try:
                original_idx = sources.index(doc) + 1
            except ValueError:
                original_idx = "?"
            
            # Handle both dict and object access for metadata
            metadata = doc.metadata if hasattr(doc, 'metadata') else doc.get('metadata', {})
            source = os.path.basename(metadata.get('source', 'Desconocido'))
            page = metadata.get('page', 0) + 1
            source_file = metadata.get('source_file', '')
            page_content = doc.page_content if hasattr(doc, 'page_content') else doc.get('page_content', '')
            
            # Button/Popover for the source
            with st.popover(f"📄 [{original_idx}] {source[:15]}...", use_container_width=True):
                st.markdown(f"**Fuente [{original_idx}]: {source}**")
                st.caption(f"Página {page}")
                
                tab1, tab2 = st.tabs(["📄 PDF", "📝 Texto"])
                
                with tab1:
                    if source_file:
                        full_path = os.path.join(settings.TEMP_DOCS_DIR, source_file)
                        if os.path.exists(full_path):
                            pdf_viewer(
                                full_path, 
                                width=700, 
                                height=600, 
                                pages_to_render=[page], 
                                render_text=True,
                                key=f"pdf_viewer_{original_idx}_{i}_{page}_{id(doc)}"
                            )
                        else:
                            st.error(f"Archivo no encontrado: {source_file}")
                    else:
                        st.warning("Ruta de archivo no disponible.")
                
                with tab2:
                    st.markdown(f"_{page_content}_")

    # Expander for all other details or if user wants to see more
    with st.expander(f"Ver todos los {len(sources)} documentos recuperados"):
        for i, doc in enumerate(sources):
            # Handle both dict and object access for metadata
            metadata = doc.metadata if hasattr(doc, 'metadata') else doc.get('metadata', {})
            source = os.path.basename(metadata.get('source', 'Desconocido'))
            page = metadata.get('page', 0) + 1
            
            st.markdown(f"**[{i+1}] {source}** (Pág. {page})")
            st.caption(f"_{doc.page_content[:150] if hasattr(doc, 'page_content') else doc.get('page_content', '')[:150]}..._")
            st.divider()
