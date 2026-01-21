import os
from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from ..dao.document_dao import DocumentDAO
from ..dao.vector_dao import VectorDAO

class IngestionPipeline:
    def __init__(self, document_dao: DocumentDAO, vector_dao: VectorDAO):
        self.document_dao = document_dao
        self.vector_dao = vector_dao
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )

    def ingest_file(self, file_name: str) -> List[str]:
        """
        Ingest a single file: Load -> Split -> Embed -> Store.
        Returns list of document IDs (or just success message).
        """
        print(f"Starting ingestion for {file_name}...")
        
        # 1. Load content
        try:
            content_bytes = self.document_dao.get_document_content(file_name)
            # Simple text decoding for simulation. 
            # In production, use appropriate LangChain loaders based on file extension.
            text_content = content_bytes.decode("utf-8")
        except UnicodeDecodeError:
            print(f"Skipping {file_name}: Could not decode as UTF-8.")
            return []
        except Exception as e:
            print(f"Error loading {file_name}: {e}")
            return []

        # 2. Create Document object
        raw_doc = Document(
            page_content=text_content,
            metadata={"source": file_name}
        )

        # 3. Split text
        chunks = self.text_splitter.split_documents([raw_doc])
        print(f"Split {file_name} into {len(chunks)} chunks.")

        # 4. Store in Vector Search (handles embedding internally in our service)
        self.vector_dao.save_vectors(chunks)
        print(f"Successfully ingested {file_name}.")
        
        return [str(i) for i in range(len(chunks))]

    def run_full_ingestion(self):
        """Scan all files in storage and ingest them."""
        files = self.document_dao.list_documents()
        print(f"Found {len(files)} files to ingest.")
        for f in files:
            self.ingest_file(f)

# Helper to easily run from main or script
def run_ingestion(storage_path: str = "data/raw", vector_path: str = "data/vector_index"):
    # Imports inside to avoid circular deps if any, or just for cleanliness in script usage
    from ..services.storage import LocalStorageService
    from ..services.embeddings import LocalEmbeddingService
    from ..services.vector_search import LocalVectorStoreService
    
    storage_svc = LocalStorageService(base_path=storage_path)
    # We need to ensure embeddings and vector store are initialized
    embedding_svc = LocalEmbeddingService()
    vector_svc = LocalVectorStoreService(embedding_service=embedding_svc, index_path=vector_path)
    
    doc_dao = DocumentDAO(storage_svc)
    vec_dao = VectorDAO(vector_svc)
    
    pipeline = IngestionPipeline(doc_dao, vec_dao)
    pipeline.run_full_ingestion()

if __name__ == "__main__":
    # Allow running this file directly to trigger ingestion
    run_ingestion()
