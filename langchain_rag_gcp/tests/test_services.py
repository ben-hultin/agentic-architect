import os
import shutil
import pytest
from langchain_rag_gcp.src.services.storage import LocalStorageService
from langchain_rag_gcp.src.services.embeddings import LocalEmbeddingService
from langchain_rag_gcp.src.services.vector_search import LocalVectorStoreService
from langchain_rag_gcp.src.services.llm import LocalGenAIService
from langchain_core.documents import Document

TEST_DATA_DIR = "tests/data_temp"

@pytest.fixture
def storage_service():
    if os.path.exists(TEST_DATA_DIR):
        shutil.rmtree(TEST_DATA_DIR)
    service = LocalStorageService(base_path=TEST_DATA_DIR)
    yield service
    if os.path.exists(TEST_DATA_DIR):
        shutil.rmtree(TEST_DATA_DIR)

def test_storage_service(storage_service):
    filename = "test.txt"
    content = b"Hello World"
    
    path = storage_service.upload_file(filename, content)
    assert os.path.exists(path)
    
    read_content = storage_service.read_file(filename)
    assert read_content == content
    
    files = storage_service.list_files()
    assert filename in files

def test_embedding_service():
    service = LocalEmbeddingService()
    embeddings = service.get_embeddings_model()
    vector = embeddings.embed_query("test")
    assert len(vector) == 768

def test_vector_store_service():
    # Setup
    embed_svc = LocalEmbeddingService()
    vector_svc = LocalVectorStoreService(embed_svc, index_path="tests/vector_temp")
    
    doc = Document(page_content="test content", metadata={"id": 1})
    vector_svc.add_documents([doc])
    
    results = vector_svc.similarity_search("test", k=1)
    assert len(results) == 1
    assert results[0].page_content == "test content"
    
    # Cleanup
    if os.path.exists("tests/vector_temp"):
        shutil.rmtree("tests/vector_temp")

def test_llm_service():
    service = LocalGenAIService(responses=["Test Response"])
    llm = service.get_llm()
    response = llm.invoke("Hello")
    assert response.content == "Test Response"
