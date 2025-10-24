from ingest import load_document, chunk_document, find_images_in_page, image_to_text
from PIL.Image import Image

def test_load_document():
    file_path = "sample.txt"
    document = load_document(file_path)
    assert isinstance(document, str)
    assert len(document) > 0

def test_load_document_invalid_path():
    file_path = "non_existent_file.txt"
    try:
        load_document(file_path)
    except FileNotFoundError:
        assert True
    else:
        assert False, "Expected FileNotFoundError"

def test_load_document_unsupported_format():
    file_path = "unsupported_file.xyz"
    try:
        load_document(file_path)
    except ValueError:
        assert True
    else:
        assert False, "Expected ValueError for unsupported format"

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

def test_find_images_in_page():
    document = "Document with images."
    images = find_images_in_page(document)
    assert isinstance(images, list)
    assert all(isinstance(image, Image) for image in images)
    assert len(images) >= 0

def test_find_images_no_images():
    document = "Page without images."
    images = find_images_in_page(document)
    assert images == []

def test_image_to_text():
    image_path = "sample_image.png"
    text = image_to_text(image_path)
    assert isinstance(text, str)
    assert len(text) > 0


