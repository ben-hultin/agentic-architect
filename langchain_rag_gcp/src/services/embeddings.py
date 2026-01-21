from abc import ABC, abstractmethod
from typing import List, Any
from langchain_core.embeddings import Embeddings
from langchain_community.embeddings import FakeEmbeddings

class EmbeddingService(ABC):
    """Abstract base class for embedding services."""
    
    @abstractmethod
    def get_embeddings_model(self) -> Embeddings:
        """Return the LangChain Embeddings model."""
        pass

class LocalEmbeddingService(EmbeddingService):
    """Local simulation of Vertex AI Embeddings using FakeEmbeddings."""
    
    def __init__(self, size: int = 768):
        self.size = size
        self.model = FakeEmbeddings(size=size)
        
    def get_embeddings_model(self) -> Embeddings:
        return self.model
