import chainlit as cl

class ChainlitFileAdapter:
    """Adapta un archivo de Chainlit para que parezca un UploadedFile de Streamlit/Python."""
    def __init__(self, cl_file: cl.File):
        self.name = cl_file.name
        self.path = cl_file.path
    
    def read(self):
        with open(self.path, "rb") as f:
            return f.read()
