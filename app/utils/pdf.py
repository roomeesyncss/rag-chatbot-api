from PyPDF2 import PdfReader
from typing import BinaryIO

def extract_text_from_pdf(pdf_file: BinaryIO) -> str:
    """Extract text from PDF file"""
    pdf_reader = PdfReader(pdf_file)
    text = ""
    
    for page in pdf_reader.pages:
        text += page.extract_text() + "\n"
    
    return text.strip()
