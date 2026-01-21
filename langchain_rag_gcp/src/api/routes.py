from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import List, Optional
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from ..services.llm import GenAIService
from ..dao.vector_dao import VectorDAO

router = APIRouter()

class ChatRequest(BaseModel):
    query: str
    
class ChatResponse(BaseModel):
    answer: str
    sources: List[str]

# Dependency injection for services will be handled via app state or closure if simple, 
# but for now let's assume they are injected or available.
# To keep it clean, we can define a get_rag_chain dependency.

def get_rag_chain(vector_dao: VectorDAO, llm_service: GenAIService):
    """
    Construct the RAG chain.
    """
    retriever = vector_dao.get_retriever(search_kwargs={"k": 3})
    llm = llm_service.get_llm()
    
    template = """Answer the question based only on the following context:
    {context}
    
    Question: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)
    
    def format_docs(docs):
        return "\n\n".join([d.page_content for d in docs])

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return chain, retriever

from ..dependencies import get_vector_dao, get_llm_service

@router.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    vector_dao: VectorDAO = Depends(get_vector_dao),
    llm_service: GenAIService = Depends(get_llm_service)
):
    chain, retriever = get_rag_chain(vector_dao, llm_service)
    
    # Run the chain
    answer = chain.invoke(request.query)
    
    # Retrieve sources for citation (re-running retrieval or capturing it would be better, 
    # but for simple RAG we can just do a similarity search to get sources explicitly 
    # if the chain doesn't return them easily without callbacks).
    # To keep it simple and accurate:
    docs = vector_dao.search_similar(request.query, k=3)
    sources = [d.metadata.get("source", "unknown") for d in docs]
    # Deduplicate sources
    sources = list(set(sources))
    
    return ChatResponse(answer=answer, sources=sources)
