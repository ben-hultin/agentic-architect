from fastapi import FastAPI
try:
    from .src.api.routes import router as chat_router
except ImportError:
    from src.api.routes import router as chat_router

app = FastAPI(title="LangChain RAG GCP Simulation")

app.include_router(chat_router, prefix="/api/v1")

@app.get("/")
async def root():
    return {
        "message": "LangChain RAG GCP Simulation API",
        "docs": "/docs",
        "chat_endpoint": "/api/v1/chat"
    }
