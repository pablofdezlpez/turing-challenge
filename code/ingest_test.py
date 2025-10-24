import pypdf
import pytest
from ingest import chunk_document, document_to_text, load_document


@pytest.fixture
def document_path():
    return "docs/sample.pdf"

def test_load_document(document_path):
    document = load_document(document_path)
    assert isinstance(document, pypdf.PdfReader)
    assert len(document.pages) > 0

def test_load_document_invalid_path():
    file_path = "non_existent_file.pdf"
    try:
        load_document(file_path)
    except FileNotFoundError:
        assert True
    else:
        assert False, "Expected FileNotFoundError"

def test_load_document_unsupported_format():
    file_path = "docs/sample.jpeg"
    try:
        load_document(file_path)
    except ValueError:
        assert True
    else:
        assert False, "Expected ValueError for unsupported format"

def test_document_to_text(mocker, document_path):
    mocked_image = "mocked image"
    func1_mock = mocker.patch("ingest.image_to_text")
    func1_mock.return_value = mocked_image
    document = load_document(document_path)
    text = document_to_text(document)
    assert isinstance(text, str)
    assert all(isinstance(line, str) for line in text.splitlines())

def test_chunk_document():
    document = "This is a sample document. " * 100
    chunks = chunk_document(document, chunk_size=50, overlap=10)
    assert isinstance(chunks, list)
    assert all(isinstance(chunk, str) for chunk in chunks)
    assert len(chunks) == len(document) // (50 - 10) + 1

def test_chunk_smaller_than_chunk_size():
    document = "Short doc."
    chunks = chunk_document(document, chunk_size=50, overlap=10)
    assert len(chunks) == 1
    assert chunks[0] == document
