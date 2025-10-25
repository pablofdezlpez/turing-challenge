import io
import pypdf

from langchain_text_splitters import RecursiveCharacterTextSplitter
from PIL import Image
from pathlib import Path

from utils import init_vector_store


def image_to_text(image: Image.Image) -> str:
    # TODO: integrate a vision model
    return "Extracted text from image"


def load_document(file_path: Path) -> pypdf.PdfReader:
    if not file_path.suffix == ".pdf":
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
    for i, page in enumerate(document.pages):
        text += page.extract_text()
        for count, image_file_object in enumerate(page.images):
            image = Image.open(io.BytesIO(image_file_object.data))
            text_image = image_to_text(image)
            text += f"\n[On page {i + 1} Image {count + 1} info]: {text_image}"

    return text


def ingest_docs(docs_path: str):
    docs_path = Path(docs_path)
    for file_path in docs_path.iterdir():
        document = load_document(file_path)

        text = document_to_text(document)
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,  # chunk size (characters)
            chunk_overlap=200,  # chunk overlap (characters)
            add_start_index=True,  # track index in original document
        )
        chunks = text_splitter.create_documents(text)
        vector_store = init_vector_store()
        document_ids = vector_store.add_documents(documents=chunks)
        return document_ids


if __name__ == "__main__":
    ingest_docs("docs/")
