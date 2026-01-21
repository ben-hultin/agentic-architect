import pytest
import os
import shutil
from langchain_rag_gcp.src.ingestion.pipeline import IngestionPipeline
from langchain_rag_gcp.src.dao.document_dao import DocumentDAO
from langchain_rag_gcp.src.dao.vector_dao import VectorDAO
from langchain_rag_gcp.src.services.storage import LocalStorageService
from langchain_rag_gcp.src.services.embeddings import LocalEmbeddingService
from langchain_rag_gcp.src.services.vector_search import LocalVectorStoreService

TEST_PIPE_DIR = "tests/pipeline_data"
TEST_VEC_DIR = "tests/pipeline_vec"

@pytest.fixture
def pipeline_setup():
    if os.path.exists(TEST_PIPE_DIR):
        shutil.rmtree(TEST_PIPE_DIR)
    if os.path.exists(TEST_VEC_DIR):
        shutil.rmtree(TEST_VEC_DIR)
        
    storage = LocalStorageService(base_path=TEST_PIPE_DIR)
    embed = LocalEmbeddingService()
    vector = LocalVectorStoreService(embed, index_path=TEST_VEC_DIR)
    
    doc_dao = DocumentDAO(storage)
    vec_dao = VectorDAO(vector)
    
    pipeline = IngestionPipeline(doc_dao, vec_dao)
    
    yield pipeline, doc_dao
    
    if os.path.exists(TEST_PIPE_DIR):
        shutil.rmtree(TEST_PIPE_DIR)
    if os.path.exists(TEST_VEC_DIR):
        shutil.rmtree(TEST_VEC_DIR)

def test_ingestion_flow(pipeline_setup):
    pipeline, doc_dao = pipeline_setup
    
    # 1. Create a file
    doc_dao.save_document("info.txt", b"This is important information about the project.")
    
    # 2. Run ingestion
    ids = pipeline.ingest_file("info.txt")
    
    assert len(ids) > 0
    
    # 3. Verify it's in vector store
    # We can check by searching via the vector dao linked to the pipeline
    results = pipeline.vector_dao.search_similar("important", k=1)
    assert len(results) == 1
    assert "important information" in results[0].page_content
