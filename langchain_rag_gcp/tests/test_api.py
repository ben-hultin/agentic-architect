from fastapi.testclient import TestClient
from langchain_rag_gcp.main import app
from langchain_rag_gcp.src.dependencies import get_vector_dao, get_llm_service
from langchain_rag_gcp.src.services.llm import LocalGenAIService
from langchain_rag_gcp.src.services.vector_search import LocalVectorStoreService
from langchain_rag_gcp.src.services.embeddings import LocalEmbeddingService
from langchain_rag_gcp.src.dao.vector_dao import VectorDAO
import pytest

client = TestClient(app)

# Override dependencies if needed, or rely on the cached local simulation
# For unit tests, it's often better to override with mocks, but since our services are local simulations, 
# we can use them directly or lightweight overrides.

def test_read_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["message"] == "LangChain RAG GCP Simulation API"

def test_chat_endpoint():
    # Ensure we have a vector store set up or it returns empty
    # We can inject a pre-populated DAO or just expect a valid empty response/mocked response.
    
    # Let's seed the LLM response to be deterministic
    def override_llm():
        return LocalGenAIService(responses=["I am a test bot."])
        
    app.dependency_overrides[get_llm_service] = override_llm
    
    response = client.post("/api/v1/chat", json={"query": "Hello"})
    assert response.status_code == 200
    data = response.json()
    assert data["answer"] == "I am a test bot."
    assert "sources" in data
