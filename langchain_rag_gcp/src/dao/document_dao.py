from typing import List, Optional
from ..services.storage import StorageService

class DocumentDAO:
    """DAO for accessing raw documents from storage."""
    
    def __init__(self, storage_service: StorageService):
        self.storage_service = storage_service
        
    def list_documents(self) -> List[str]:
        """List all available documents."""
        return self.storage_service.list_files()
        
    def get_document_content(self, file_name: str) -> bytes:
        """Get raw content of a document."""
        return self.storage_service.read_file(file_name)
    
    def save_document(self, file_name: str, content: bytes) -> str:
        """Save a document to storage."""
        return self.storage_service.upload_file(file_name, content)
