# LangChain RAG GCP Simulation

This project implements a Retrieval-Augmented Generation (RAG) system following the Google Cloud Platform (GCP) architecture but running entirely locally with simulated services. It demonstrates the ingestion and serving pipelines typical of a cloud-native GenAI application without requiring live cloud connectivity.

## Overview

The system is designed with a modular architecture that mirrors a production GCP setup:
- **Ingestion Pipeline**: Processes raw documents, chunks them, and stores embeddings.
- **Serving Pipeline**: A FastAPI backend that handles user queries, retrieves relevant context, and generates responses.
- **Simulation**: Service layers (Storage, Vector Search, LLM) are abstracted, with local file-based implementations provided for offline development.

## Project Structure

```
langchain_rag_gcp/
├── data/
│   ├── raw/                 # Place raw text documents here
│   └── vector_index/        # FAISS vector store persistence
├── src/
│   ├── api/                 # FastAPI routes and server logic
│   ├── dao/                 # Data Access Objects
│   ├── ingestion/           # Document processing pipeline
│   ├── services/            # Service interfaces and local simulations
│   ├── models/              # Pydantic data models
│   └── dependencies.py      # Dependency injection setup
├── tests/                   # Unit tests
├── main.py                  # Application entry point
└── requirements.txt         # Project dependencies
```

## Setup & Usage

1.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

2.  **Add Documents**
    Place `.txt` files in the `data/raw/` directory. (Sample files are included).

3.  **Run Ingestion**
    Process the documents and build the vector index:
    ```bash
    python -m src.ingestion.pipeline
    ```

4.  **Start the API Server**
    Launch the FastAPI backend:
    ```bash
    uvicorn main:app --reload
    ```
    The API will be available at `http://127.0.0.1:8000`.

5.  **Test the Chat**
    Send a query to the chat endpoint:
    ```bash
    curl -X POST "http://127.0.0.1:8000/api/v1/chat" \
         -H "Content-Type: application/json" \
         -d '{"query": "What is RAG?"}'
    ```

## Architecture Layers

-   **Storage**: Simulates Google Cloud Storage using a local `data/raw` directory.
-   **Embeddings**: Uses `FakeEmbeddings` (LangChain) to simulate Vertex AI Embeddings.
-   **Vector Search**: Uses `FAISS` to simulate Vertex AI Vector Search.
-   **LLM**: Uses `FakeListChatModel` (LangChain) to simulate Gemini, returning mock responses.

## Testing

Run the test suite with `pytest`:
```bash
pytest tests/
```
