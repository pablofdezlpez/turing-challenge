
def find_images_in_page(document):
    pass

def load_document(file_path):
    pass

def chunk_document(document, chunk_size=1000, overlap=200):
    pass

def image_to_text(image_path):
    pass

def ingest_file(file_path):
    document = load_document(file_path)
    chunks = chunk_document(document)
    vector_store = None
    images = find_images_in_page(document)
    for image in images:
        text = image_to_text(image)
        chunks.add_vectors([text])

    document_ids = vector_store.add_documents(documents=chunks)
    return document_ids

if __name__ == "__main__":
    file_path = "path/to/document"
    ingest_file(file_path)