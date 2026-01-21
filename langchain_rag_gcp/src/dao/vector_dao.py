from typing import List, Any
from langchain_core.documents import Document
from ..services.vector_search import VectorSearchService

class VectorDAO:
    """DAO for interacting with the vector store."""
    
    def __init__(self, vector_service: VectorSearchService):
        self.vector_service = vector_service
        
    def save_vectors(self, documents: List[Document]) -> None:
        """Save document embeddings to vector store."""
        self.vector_service.add_documents(documents)
        
    def search_similar(self, query: str, k: int = 4) -> List[Document]:
        """Search for similar documents."""
        return self.vector_service.similarity_search(query, k)
        
    def get_retriever(self, **kwargs) -> Any:
        """Get a retriever for the vector store."""
        return self.vector_service.get_retriever(**kwargs)
