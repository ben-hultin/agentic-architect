from abc import ABC, abstractmethod
from typing import List, Any, Optional
import os
from langchain_core.vectorstores import VectorStore
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from .embeddings import EmbeddingService

class VectorSearchService(ABC):
    """Abstract base class for vector search services."""
    
    @abstractmethod
    def add_documents(self, documents: List[Document]) -> None:
        """Add documents to the vector store."""
        pass
        
    @abstractmethod
    def similarity_search(self, query: str, k: int = 4) -> List[Document]:
        """Search for similar documents."""
        pass
        
    @abstractmethod
    def get_retriever(self, **kwargs) -> Any:
        """Return a LangChain retriever interface."""
        pass

class LocalVectorStoreService(VectorSearchService):
    """Local simulation of Vector Search using FAISS."""
    
    def __init__(self, embedding_service: EmbeddingService, index_path: str = "data/vector_index"):
        self.embedding_service = embedding_service
        self.index_path = index_path
        self.embeddings = embedding_service.get_embeddings_model()
        self.vector_store: Optional[FAISS] = None
        self._load_or_create_index()

    def _load_or_create_index(self):
        if os.path.exists(self.index_path):
            try:
                self.vector_store = FAISS.load_local(
                    self.index_path, 
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )
            except Exception:
                # Fallback if load fails or file is corrupt
                pass
        
        # If still None, it means we didn't load it
        # We can't initialize FAISS without documents usually, 
        # but we can initialize an empty one with a trick or just wait for first add.
        # Actually FAISS.from_texts([""], embeddings) works to init.
        if self.vector_store is None:
            # Initialize with a dummy doc to create structure, then we might need to clear it or just ignore.
            # Better approach: store is None until docs are added? 
            # Or just create a fresh one in memory if not persistent.
            # For simplicity in simulation, let's allow in-memory start.
            pass

    def add_documents(self, documents: List[Document]) -> None:
        if not documents:
            return
            
        if self.vector_store is None:
            self.vector_store = FAISS.from_documents(documents, self.embeddings)
        else:
            self.vector_store.add_documents(documents)
            
        # Save local index
        if self.vector_store:
            self.vector_store.save_local(self.index_path)

    def similarity_search(self, query: str, k: int = 4) -> List[Document]:
        if self.vector_store is None:
            return []
        return self.vector_store.similarity_search(query, k=k)

    def get_retriever(self, **kwargs) -> Any:
        if self.vector_store is None:
            # If empty, return a dummy retriever or raise
            # Create a temporary empty store to return a retriever that returns nothing
            empty_store = FAISS.from_texts([" "], self.embeddings)
            return empty_store.as_retriever(**kwargs)
        return self.vector_store.as_retriever(**kwargs)
