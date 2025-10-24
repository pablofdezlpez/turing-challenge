import io

import pypdf
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from PIL import Image

load_dotenv()

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


if __name__ == "__main__":
    file_path = "docs/sample.pdf"
    document = load_document(file_path)
      
    text = document_to_text(document)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,  # chunk size (characters)
        chunk_overlap=200,  # chunk overlap (characters)
        add_start_index=True,  # track index in original document
    )
    chunks = text_splitter.create_documents(text)
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    vector_store = Chroma(
        collection_name="example_collection",
        embedding_function=embeddings,
        persist_directory="./chroma_langchain_db",
    )
    document_ids = vector_store.add_documents(documents=chunks)
