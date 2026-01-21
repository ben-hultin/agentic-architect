import pytest
from langchain_rag_gcp.src.dao.document_dao import DocumentDAO
from langchain_rag_gcp.src.dao.vector_dao import VectorDAO
from langchain_rag_gcp.src.services.storage import LocalStorageService
from langchain_rag_gcp.src.services.embeddings import LocalEmbeddingService
from langchain_rag_gcp.src.services.vector_search import LocalVectorStoreService
import shutil
import os

TEST_DAO_DIR = "tests/dao_temp"

@pytest.fixture
def doc_dao():
    if os.path.exists(TEST_DAO_DIR):
        shutil.rmtree(TEST_DAO_DIR)
    storage = LocalStorageService(base_path=TEST_DAO_DIR)
    return DocumentDAO(storage)

@pytest.fixture
def vec_dao():
    embed = LocalEmbeddingService()
    vector = LocalVectorStoreService(embed, index_path="tests/vec_dao_temp")
    return VectorDAO(vector)

def test_document_dao(doc_dao):
    doc_dao.save_document("doc1.txt", b"content")
    assert "doc1.txt" in doc_dao.list_documents()
    assert doc_dao.get_document_content("doc1.txt") == b"content"
    
    # Cleanup
    if os.path.exists(TEST_DAO_DIR):
        shutil.rmtree(TEST_DAO_DIR)

def test_vector_dao(vec_dao):
    from langchain_core.documents import Document
    doc = Document(page_content="dao test", metadata={"source": "test"})
    vec_dao.save_vectors([doc])
    
    res = vec_dao.search_similar("dao", k=1)
    assert len(res) > 0
    assert res[0].page_content == "dao test"
    
    # Cleanup
    if os.path.exists("tests/vec_dao_temp"):
        shutil.rmtree("tests/vec_dao_temp")
