"""
PDF processing utilities
"""
from io import StringIO
from pdfminer.high_level import extract_text_to_fp

def extract_text_from_pdf(filepath: str) -> str:
    """Extract text from PDF file using pdfminer"""
    output = StringIO()
    with open(filepath, 'rb') as file:
        extract_text_to_fp(file, output)
    return output.getvalue()

if __name__ == "__main__":
    # Test function
    print("PDF processing utility loaded")
