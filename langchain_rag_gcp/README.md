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

## Service Simulation Breakdown

This project simulates the following GCP services to allow for offline development:

| GCP Service | Simulated By | Role in Project |
| :--- | :--- | :--- |
| **Cloud Storage (GCS)** | `LocalStorageService` | **Data Lake**: Acts as the initial landing zone for raw documents (e.g., PDFs, text files) before they are processed. |
| **Vertex AI Embeddings** | `LocalEmbeddingService` | **Vectorization**: Converts text chunks into numerical vectors (embeddings) so they can be compared mathematically. |
| **Vertex AI Vector Search** | `LocalVectorStoreService` | **Retrieval Engine**: Stores the generated vectors and performs similarity searches to find the most relevant document chunks for a user query. |
| **Vertex AI (Gemini)** | `LocalGenAIService` | **Reasoning Engine**: The LLM that takes the user's question and the retrieved context to generate a natural language answer. |

## RAG Workflow

The system operates in two distinct pipelines:

### 1. Ingestion Pipeline (Offline)
*Goal: Prepare the knowledge base.*
1.  **Load**: Scans the "Cloud Storage" bucket (`data/raw/`) for new files.
2.  **Split**: Reads file content and splits it into smaller chunks (e.g., 1000 chars).
3.  **Embed**: Converts each text chunk into a vector representation using the Embedding Service.
4.  **Store**: Saves vectors and metadata into the "Vector Search" index (`data/vector_index/`).

### 2. Serving Pipeline (Online)
*Goal: Answer user questions.*
1.  **Receive Query**: API receives a user question (e.g., "What is RAG?").
2.  **Embed Query**: Converts the question into a vector.
3.  **Retrieve**: Finds the top $k$ most similar document chunks from the Vector Search index.
4.  **Augment**: Constructs a prompt containing the user's question and the retrieved context.
5.  **Generate**: Sends the prompt to the LLM to generate a grounded answer.

## Testing

Run the test suite with `pytest`:
```bash
pytest tests/
```
