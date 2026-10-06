# RAG FastAPI MongoDB

A lightweight, end-to-end **Retrieval-Augmented Generation (RAG)** application built with **Python, FastAPI, OpenAI, and MongoDB Atlas**.

The project processes PDF documents, converts them into structured chunks, generates embeddings, stores the knowledge in MongoDB, retrieves relevant information using hybrid search, and uses an LLM to generate answers grounded in the retrieved content.

The implementation intentionally avoids heavyweight RAG frameworks such as LangChain. Core RAG functionality is implemented with straightforward Python services and clearly separated application layers.

## Features

- **PDF ingestion and intelligent chunking**
  - Extracts text from PDF documents.
  - Uses OCR fallback for scanned/image-based pages.
  - Preserves page and section information.
  - Creates deterministic chunk identifiers.

- **OpenAI embeddings**
  - Generates vector embeddings using the configured OpenAI embedding model.
  - Stores embeddings alongside chunk metadata in MongoDB.

- **MongoDB Atlas vector and text search**
  - Uses MongoDB as the document and vector store.
  - Supports semantic vector retrieval.
  - Supports lexical/keyword retrieval.
  - Combines both using hybrid ranking.

- **RAG-based chat**
  - Retrieves relevant document passages before generation.
  - Uses the LLM to answer questions based on retrieved evidence.
  - Returns citations pointing back to source chunks/pages.
  - Supports conversation IDs for multi-turn chat.

- **Grounding and hallucination control**
  - Validates generated citations against retrieved evidence.
  - Can refuse to answer when sufficient evidence is unavailable.
  - Uses bounded retries around LLM reasoning/generation stages.

- **Multi-query retrieval**
  - Supports original and reformulated search queries.
  - Improves retrieval for questions that can be expressed in multiple ways.

- **Evaluation and testing**
  - Includes automated tests for chunking, retrieval, RAG behavior, retries, grounding, and API functionality.
  - Includes evaluation examples based on the supplied healthcare-program brochure.

## Architecture

```text
PDF Documents
     │
     ▼
PDF Extraction / OCR
     │
     ▼
Structured Chunking
     │
     ▼
OpenAI Embeddings
     │
     ▼
MongoDB Atlas
     │
     ├── Vector Search
     └── Keyword Search
             │
             ▼
        Hybrid Retrieval
             │
             ▼
        LLM / RAG Layer
             │
             ▼
      Grounded Answer
        + Citations
```

## Main Technology Stack

| Component | Technology |
|---|---|
| Language | Python |
| API | FastAPI |
| Validation | Pydantic |
| LLM | OpenAI |
| Embeddings | OpenAI Embeddings |
| Database | MongoDB Atlas |
| Vector Search | MongoDB Atlas Vector Search |
| Keyword Search | MongoDB Atlas Search |
| PDF Processing | PyMuPDF |
| OCR | Tesseract + pytesseract |
| Package Management | uv |
| Testing | pytest / pytest-asyncio |

## Main API Endpoints

### `POST /ingest`

Processes a PDF document, creates structured chunks, generates embeddings, and stores the resulting knowledge in MongoDB.

### `POST /ingest/preview`

Runs the document extraction/chunking workflow in preview mode so the generated chunks can be inspected before relying on them for retrieval.

### `POST /search`

Performs hybrid retrieval against the MongoDB knowledge base.

The endpoint can use:

- semantic/vector search
- keyword search
- multiple queries
- optional filters
- configurable `top_k`

### `POST /chat`

Runs the complete RAG workflow:

```text
User Question
     ↓
Search Planning / Query Reformulation
     ↓
Hybrid Retrieval
     ↓
Context Construction
     ↓
LLM Generation
     ↓
Grounding + Citation Validation
     ↓
Final Response
```

The endpoint supports `conversation_id` so follow-up questions can use previous conversational context.

## Configuration

Runtime configuration is supplied through environment variables.

A typical `.env` configuration contains values for:

```text
OPENAI_API_KEY
OPENAI_EMBEDDING_MODEL
OPENAI_EMBEDDING_DIMENSIONS

MONGODB_URI
MONGODB_DB
MONGODB_CHUNKS_COLLECTION
MONGODB_CONVERSATIONS_COLLECTION

MONGODB_VECTOR_INDEX
MONGODB_SEARCH_INDEX
```

Secrets should remain in the local `.env` file and must not be committed to version control.

## Running the Project

Install dependencies:

```bash
uv sync
```

Run the test suite:

```bash
uv run pytest -q
```

Start the FastAPI application:

```bash
uv run uvicorn app.main:app --reload
```

The interactive API documentation is then available at:

```text
http://127.0.0.1:8000/docs
```

## MongoDB Search Indexes

The MongoDB collection requires the appropriate Atlas Search and Vector Search indexes for retrieval.

The project includes scripts for checking and creating the required indexes.

```bash
uv run python scripts/check_mongodb.py
```

```bash
uv run python scripts/create_indexes.py
```

The Vector Search index is configured for the embedding field and the configured embedding dimensions, while the text index supports lexical retrieval.

## Project Structure

```text
app/
├── api/
│   ├── chat.py
│   ├── ingest.py
│   └── search.py
├── db/
│   ├── client.py
│   ├── collections.py
│   └── indexes.py
├── llm/
│   └── openai_client.py
├── models/
│   └── schemas.py
├── services/
│   ├── chunking.py
│   ├── embeddings.py
│   ├── pdf_loader.py
│   ├── rag.py
│   └── retrieval.py
└── main.py

scripts/
├── check_mongodb.py
├── create_indexes.py
└── preview_pdf.py

tests/
├── test_chunking.py
├── test_retrieval.py
└── test_rag.py
```

## Design Philosophy

The project is designed as a **simple, understandable RAG implementation** rather than a production-scale platform.

The main goals are:

- clear separation between ingestion, retrieval, and generation
- explicit data models
- minimal framework abstraction
- source-grounded answers
- predictable behavior and testability
- straightforward upgrade paths for future infrastructure

The LLM provider interaction is kept behind a dedicated boundary so additional observability, model-routing, or provider infrastructure can be introduced later without restructuring the entire RAG pipeline.

## Current Scope

This repository is suitable for demonstrating and evaluating a complete document-based RAG workflow using FastAPI and MongoDB Atlas.

It is intentionally kept focused on the core RAG system rather than production infrastructure such as distributed workers, advanced observability platforms, complex agent orchestration, or large-scale deployment architecture.