from functools import lru_cache
from .services.storage import LocalStorageService
from .services.embeddings import LocalEmbeddingService
from .services.vector_search import LocalVectorStoreService
from .services.llm import LocalGenAIService
from .dao.document_dao import DocumentDAO
from .dao.vector_dao import VectorDAO

# Singletons
@lru_cache()
def get_storage_service():
    return LocalStorageService()

@lru_cache()
def get_embedding_service():
    return LocalEmbeddingService()

@lru_cache()
def get_vector_service():
    # Depends on embedding service
    embed_svc = get_embedding_service()
    return LocalVectorStoreService(embedding_service=embed_svc)

@lru_cache()
def get_llm_service():
    return LocalGenAIService()

@lru_cache()
def get_document_dao():
    return DocumentDAO(get_storage_service())

@lru_cache()
def get_vector_dao():
    return VectorDAO(get_vector_service())
