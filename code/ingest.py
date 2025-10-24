import pypdf
from PIL import Image
import io

def image_to_text(image: Image.Image) -> str:
    #TODO: integrate a vision model
    return "Extracted text from image"

def load_document(file_path: str) -> pypdf.PdfReader:
    if not file_path.endswith('.pdf'):
        raise ValueError("Unsupported file format")
    document = pypdf.PdfReader(file_path)
    return document

def document_to_text(document: pypdf.PdfReader) -> str:
    """Extract the text from all pages in the document.
    Add description for images found in the document.

    Args:
        document (pypdf.PdfReader): The PDF document to extract text from.

    Returns:
        str: full text
    """
    text = ""
    for page in document.pages:
        text += page.extract_text()
        for count, image_file_object in enumerate(page.images):
            image = Image.open(io.BytesIO(image_file_object.data))
            text_image = image_to_text(image)
            text += f"\n[Image {count + 1} info]: {text_image}"
        
    return text

def chunk_document(document: str, chunk_size: int = 1000, overlap: int = 200) -> list[str]:
    """Chunk the document into smaller pieces."""
    if len(document) <= chunk_size:
        return [document]
    chunks = []
    start = 0
    while start < len(document):
        end = min(start + chunk_size, len(document))
        chunks.append(document[start:end])
        start += chunk_size - overlap
    return chunks

def ingest_file(file_path: str) -> list[str]:
    document = load_document(file_path)
    chunks = chunk_document(document)
    vector_store = None
    document_ids = vector_store.add_documents(documents=chunks)
    return document_ids

if __name__ == "__main__":
    file_path = "docs/sample.pdf"
    document = load_document(file_path)
    pages = document_to_text(document)
    chunks = chunk_document(" ".join(pages))