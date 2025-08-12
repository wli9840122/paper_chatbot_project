import os
from PyPDF2 import PdfReader


def extract_text_from_pdf(file_path: str) -> str:
    """
    Extract text content from a PDF file.

    Args:
        file_path (str): The path to the PDF file.

    Returns:
        str: The extracted text content from the PDF file, with pages concatenated.

    Raises:
        FileNotFoundError: If the specified PDF file does not exist.
    """
    if not os.path.exists(file_path):
        # Raise an error if the file does not exist
        raise FileNotFoundError(f"PDF 文件不存在: {file_path}")

    # Initialize a PDF reader for the given file
    reader = PdfReader(file_path)

    # Extract text from each page and concatenate it with line breaks
    text = "\n".join(page.extract_text() for page in reader.pages if page.extract_text())

    # Return the extracted text, stripped of leading and trailing whitespace
    return text.strip()
