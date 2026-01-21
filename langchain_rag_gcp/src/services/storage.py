from abc import ABC, abstractmethod
import os
from typing import List, Optional

class StorageService(ABC):
    """Abstract base class for storage services."""

    @abstractmethod
    def list_files(self) -> List[str]:
        """List files in the storage."""
        pass

    @abstractmethod
    def upload_file(self, file_name: str, content: bytes) -> str:
        """Upload a file to storage. Returns the path/uri."""
        pass

    @abstractmethod
    def read_file(self, file_name: str) -> bytes:
        """Read a file from storage."""
        pass

class LocalStorageService(StorageService):
    """Local file system implementation of StorageService."""

    def __init__(self, base_path: str = "data/raw"):
        self.base_path = base_path
        # Ensure the directory exists
        os.makedirs(self.base_path, exist_ok=True)

    def list_files(self) -> List[str]:
        return [f for f in os.listdir(self.base_path) if os.path.isfile(os.path.join(self.base_path, f))]

    def upload_file(self, file_name: str, content: bytes) -> str:
        file_path = os.path.join(self.base_path, file_name)
        with open(file_path, "wb") as f:
            f.write(content)
        return file_path

    def read_file(self, file_name: str) -> bytes:
        file_path = os.path.join(self.base_path, file_name)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File {file_name} not found in {self.base_path}")
        with open(file_path, "rb") as f:
            return f.read()
