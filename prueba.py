from fpdf import FPDF

def create_chaos_pdf():
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=11)

    # Título
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, "PROTOCOLO DE SEGURIDAD QUÍMICA (Estructura Caótica)", 0, 1, 'C')
    pdf.ln(10)

    # Configuración de columnas visuales
    col_left_x = 10
    col_right_x = 110
    y_start = pdf.get_y()
    line_height = 6

    # TEXTO A (Columna Izquierda): Procedimiento para ÁCIDO
    text_acid = [
        "MANEJO DE ÁCIDOS (PELIGRO):",
        "1. Nunca agregar agua al ácido.",
        "2. Usar guantes de nitrilo grueso.",
        "3. En caso de derrame, neutralizar",
        "   con bicarbonato de sodio.",
        "4. Almacenar en estante INFERIOR."
    ]

    # TEXTO B (Columna Derecha): Procedimiento para BASES
    text_base = [
        "MANEJO DE BASES (ALCALINOS):",
        "1. Se puede diluir con agua.",
        "2. Usar guantes de látex simple.",
        "3. En caso de derrame, usar vinagre",
        "   (ácido acético) diluido.",
        "4. Almacenar en estante SUPERIOR."
    ]

    # --- LA TRAMPA ---
    # Escribimos las líneas intercaladas (Línea 1 Izq, Línea 1 Der, Línea 2 Izq...)
    # Esto confunde a los loaders básicos que leen por coordenada Y (altura)
    
    pdf.set_font("Arial", size=10)
    
    for i in range(len(text_acid)):
        # Escribir línea columna izquierda
        pdf.set_xy(col_left_x, y_start + (i * line_height))
        pdf.cell(90, line_height, text_acid[i], border=1) # Borde para ver que son celdas distintas
        
        # Escribir línea columna derecha (inmediatamente después en el flujo)
        pdf.set_xy(col_right_x, y_start + (i * line_height))
        pdf.cell(90, line_height, text_base[i], border=1)

    pdf.output("validacion_iso_chaos.pdf")
    print("✅ Archivo 'validacion_iso_chaos.pdf' generado. ¡Suerte!")

if __name__ == "__main__":
    create_chaos_pdf()