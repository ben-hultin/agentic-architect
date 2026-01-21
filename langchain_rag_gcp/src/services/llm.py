from abc import ABC, abstractmethod
from typing import Any
from langchain_core.language_models import BaseChatModel
from langchain_community.chat_models import FakeListChatModel

class GenAIService(ABC):
    """Abstract base class for Generative AI services."""
    
    @abstractmethod
    def get_llm(self) -> BaseChatModel:
        """Return the LangChain Chat Model."""
        pass

class LocalGenAIService(GenAIService):
    """Local simulation of Gemini using FakeListChatModel."""
    
    def __init__(self, responses: list[str] = None):
        if responses is None:
            responses = [
                "This is a simulated response based on the retrieved context.",
                "I found some relevant information in the documents.",
                "According to the context, the answer is... (simulated)"
            ]
        self.model = FakeListChatModel(responses=responses)
        
    def get_llm(self) -> BaseChatModel:
        return self.model
