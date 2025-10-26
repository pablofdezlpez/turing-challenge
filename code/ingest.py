import io
import pypdf

from langchain_text_splitters import RecursiveCharacterTextSplitter
from PIL import Image
from pathlib import Path
from pydantic import BaseModel
from langchain.chat_models import BaseChatModel
from utils import init_vector_store, init_chat_llm, image_to_base64


class StructuredOutput(BaseModel):
    name: str
    age: int
    role: str
    years_of_experience: int


def extract_structured_data(text: str, llm: BaseChatModel) -> StructuredOutput:
    response = llm.invoke([
            {'role': 'system', 'content': 'Extract the following fields from the given CV text: name, age, role, years_of_experience. Return the response in a structured format.'},
            {"role": "user", "content": text}
        ]
    )

    return response


def image_to_text(image: Image.Image, vision_llm: BaseChatModel) -> str:
    description = vision_llm.invoke([
        {
            "role": "user",
            "content": [
                    {"type": "text", "text": "Describe the content of this image."},
                    {
                        "type": "image", 
                        "source_type": "base64",
                        "mime_type": "image/png",
                        "data": image_to_base64(image).decode('utf-8')
                    }
                ]
            }
            
        ],
    )
    return description


def load_document(file_path: Path) -> pypdf.PdfReader:
    if not file_path.suffix == ".pdf":
        raise ValueError("Unsupported file format")
    document = pypdf.PdfReader(file_path)
    return document


def document_to_text(document: pypdf.PdfReader, llm: BaseChatModel) -> str:
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
            text_image = image_to_text(image, llm)
            text += f"\n[On page {i + 1} Image {count + 1} info]: {text_image}"
    return text


def ingest_docs(docs_path: str, llm_model: str = "gpt-5-nano"):
    llm = init_chat_llm(llm_model, use_tools=False)
    llm_with_structured_output = llm.with_structured_output(StructuredOutput)
    docs_path = Path(docs_path)
    for file_path in docs_path.iterdir():
        print(f"Ingesting document: {file_path.name}")
        document = load_document(file_path)

        # Transform document to text, split into chunks, and ingest into vector store
        text = document_to_text(document, llm)
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,  # chunk size (characters)
            chunk_overlap=200,  # chunk overlap (characters)
            add_start_index=True,  # track index in original document
        )
        chunks = text_splitter.create_documents(text)
        vector_store = init_vector_store()
        document_ids = vector_store.add_documents(documents=chunks)

        # If document is a CV, extract structured data
        if "cv" in file_path.stem.lower():
            structured_data = extract_structured_data(text, llm_with_structured_output)
            print(f"Extracted structured data from {file_path.name}: {structured_data}")
    return document_ids


if __name__ == "__main__":
    ingest_docs("docs/")
