# RAG Project — In-Depth Technical Documentation

## 1. Purpose and final scope

This document describes the latest state of the **FastAPI + MongoDB RAG assessment project** in technical detail: architecture, file responsibilities, data contracts, ingestion, OCR, chunking, embeddings, MongoDB storage and indexes, retrieval, RRF ranking, query reformulation, multi-hop handling, conversation continuity, grounded generation, fail-closed validation, retries, evaluation, tests, operational debugging, and known limitations.

The objective was not to build a production platform. The objective was to implement a **small but robust end-to-end RAG system** that is understandable from source code and satisfies the assignment requirements without depending on LangChain or another RAG framework.

---

## 2. High-level architecture

```text
                                  ┌───────────────────────────┐
                                  │       FastAPI API         │
                                  │ /health /ingest /search  │
                                  │        /chat              │
                                  └─────────────┬─────────────┘
                                                │
                           ┌────────────────────┴────────────────────┐
                           │                                         │
                           ▼                                         ▼
                  ┌──────────────────┐                     ┌──────────────────┐
                  │  Ingestion flow │                     │     Chat flow    │
                  └────────┬─────────┘                     └────────┬─────────┘
                           │                                        │
             ┌─────────────▼─────────────┐              ┌───────────▼───────────┐
             │ PyMuPDF native extraction │              │ Conversation history │
             └─────────────┬─────────────┘              └───────────┬───────────┘
                           │                                        │
                  native text usable?                              │
                      │          │                                  ▼
                     yes        no                         ┌────────────────────┐
                      │          │                         │ OpenAI search-plan │
                      │          ▼                         │ tool/function call │
                      │   ┌──────────────┐                └─────────┬──────────┘
                      │   │ Tesseract OCR│                          │
                      │   └──────┬───────┘                          ▼
                      └──────────┴─────────────┐          ┌────────────────────┐
                                               ▼          │ Hybrid retrieval   │
                                      ┌────────────────┐   │ vector + keyword   │
                                      │ Structure-aware│   └─────────┬──────────┘
                                      │ chunking       │             │
                                      └───────┬────────┘             ▼
                                              │             ┌────────────────────┐
                                              ▼             │ RRF fused evidence │
                                      ┌────────────────┐   └─────────┬──────────┘
                                      │ OpenAI embedder│             │
                                      └───────┬────────┘             ▼
                                              │             ┌────────────────────┐
                                              ▼             │ Final grounded LLM │
                                      ┌────────────────┐   │ structured output  │
                                      │ MongoDB Atlas  │   └─────────┬──────────┘
                                      │ chunks         │             │
                                      └────────────────┘             ▼
                                                           ┌────────────────────┐
                                                           │ Grounding validator│
                                                           └─────────┬──────────┘
                                                                     │
                                                              valid / invalid
                                                                     │
                                               ┌─────────────────────┴───────────────────┐
                                               ▼                                         ▼
                                   grounded answer + citations                    safe abstention
```

### Architectural boundaries

There are four deliberately clear layers:

```text
API layer
  → Pydantic models
  → application services
  → database repositories / search pipelines
  → provider adapter
```

The provider-specific OpenAI SDK code lives only in `app/llm/openai_client.py`.

That is the seam intended for a future LiteLLM/Langfuse migration.

---

## 3. Repository structure and responsibilities

```text
app/
├── api/
│   ├── ingest.py
│   ├── search.py
│   └── chat.py
├── core/
│   └── config.py
├── db/
│   ├── mongodb.py
│   ├── indexes.py
│   ├── chunk_repository.py
│   └── search_pipelines.py
├── llm/
│   └── openai_client.py
├── models/
│   └── schemas.py
└── services/
    ├── pdf_loader.py
    ├── chunking.py
    ├── embedding_service.py
    ├── retrieval.py
    ├── conversation.py
    ├── rag.py
    └── evaluation.py

scripts/
├── preview_pdf.py
├── create_indexes.py
├── check_mongodb.py
└── evaluate.py

tests/
└── 15 tests across chunking, embeddings, retrieval, MongoDB/index behavior, RAG validation, and evaluation
```

### `app/models/schemas.py`

This file defines the typed contracts used throughout the application.

Important models:

- `PageText` — one extracted page and its extraction method.
- `Chunk` — structured chunk before embeddings.
- `EmbeddedChunk` — chunk plus embedding vector metadata.
- `SearchRequest` — API request for retrieval.
- `SearchFilters` — optional document/page/section constraints.
- `SearchToolArgs` — strict tool schema returned by the LLM search planner.
- `RetrievedChunk` — normalized retrieval result.
- `SearchResponse` — `/search` response.
- `Citation` — `(chunk_id, page_number)` evidence reference.
- `ChatRequest` / `ChatResponse` — chat API contract.
- `GroundedAnswer` — strict structured LLM response.

`extra="forbid"` is used on these models to prevent silent acceptance of unexpected fields. This matters especially for the LLM boundary because the downstream code should not operate on an unknown payload shape.

---

## 4. Configuration

`app/core/config.py` uses `pydantic-settings` to load environment variables from `.env`.

The important defaults are:

```text
OPENAI_EMBEDDING_MODEL      = text-embedding-3-large
OPENAI_EMBEDDING_DIMENSIONS = 3072
OPENAI_CHAT_MODEL           = gpt-6.1-sol
MONGODB_DB                  = rag_demo
MONGODB_CHUNKS_COLLECTION   = chunks
MONGODB_VECTOR_INDEX        = chunk_vector_index
MONGODB_SEARCH_INDEX        = chunk_text_index
```

The project has two explicit “required credential” checks:

```python
settings.require_openai()
settings.require_mongodb()
```

They fail early with a clear configuration error instead of creating a client with a missing credential.

### Why keep configuration in one place?

Because the same settings must be used consistently by:

- ingestion;
- embedding generation;
- MongoDB storage;
- vector retrieval;
- keyword retrieval;
- LLM generation;
- index creation scripts.

Without one configuration source, it is easy to index into one database/collection while searching another.

---

## 5. PDF ingestion and OCR

### Endpoint: `POST /ingest/preview`

Implemented in `app/api/ingest.py`.

The endpoint performs:

```text
UploadFile
→ read bytes
→ deterministic document_id
→ extract_pages()
→ chunk_pages()
→ return sample chunks
```

It intentionally stops before embeddings and persistence.

### Why preview exists

Embedding every chunk before checking extraction quality is wasteful. Preview allows inspection of:

- number of pages;
- how many pages use native extraction;
- how many pages use OCR;
- total chunk count;
- representative sample chunks.

### Document ID

The PDF bytes are hashed with SHA-256 and the first 16 bytes are represented as a deterministic UUID-shaped document ID.

Conceptually:

```python
sha256(pdf_bytes) → first 16 bytes → UUID
```

This means the same PDF content maps to the same document ID, which helps make ingestion deterministic and idempotent.

### `app/services/pdf_loader.py`

The extraction strategy is:

1. Open the PDF with PyMuPDF.
2. Attempt native text extraction.
3. If the extracted text is below the configured usable threshold, render the page to an image.
4. Run Tesseract OCR.
5. Reconstruct text lines using OCR word bounding boxes.
6. Return `PageText` with extraction method and OCR confidence.

The relevant constants include:

```text
NATIVE_TEXT_MIN_CHARS = 40
OCR_DPI               = 180
```

### Why OCR confidence is tracked

OCR is inherently less reliable than native text extraction. The loader keeps an average confidence score for each OCR page so that extraction quality is observable.

The system deliberately **does not auto-correct OCR numbers**. That is an important hallucination-control decision for an assessment involving placement statistics.

If OCR produces a questionable value, the system should preserve the extracted evidence rather than silently “fixing” it from assumptions.

### Tesseract is an OS dependency

`pytesseract` is the Python interface. Tesseract itself must be installed on the machine.

Debian/Ubuntu:

```bash
sudo apt update
sudo apt install -y tesseract-ocr
```

Without this, both `/ingest/preview` and `/ingest` can fail when OCR is invoked.

---

## 6. Structure-aware chunking

Implemented in `app/services/chunking.py`.

The chunker was intentionally built without a RAG framework.

### Chunking targets

```text
TARGET_TOKENS  = 420
MAX_TOKENS     = 520
OVERLAP_TOKENS = 60
```

The intended behavior is:

- prefer chunks around 420 tokens;
- never exceed the 520-token hard ceiling;
- use overlap when a large text block must be split.

The implementation includes a post-check so that the published `token_count()` agrees with the final piece.

### Why structure-aware instead of fixed-size splitting?

The brochure contains meaningful document structure:

- headings;
- faculty names;
- bullets;
- program sections;
- placement headings;
- page-level groupings.

A blind fixed-length splitter can separate:

```text
Placement Facts (2021-2023)
```

from its corresponding numbers, or separate a faculty name from their role.

The chunker instead tracks a `section_path` and includes that context in `embedding_text`.

Example conceptual chunk:

```text
section_path:
  ["Faculties of Healthcare Program"]

content:
  Dr. Anjali Kumar
  Associate Professor & Program InCharge - Healthcare Management
```

`embedding_text` combines section context with chunk content. This makes semantic indexing aware of the section even when the page content itself is fragmented.

### Heading detection

The chunker uses lightweight regex heuristics for known document patterns, such as:

- `USPs of the Program`
- `Faculties of Healthcare Program`
- `Industry Engagement`
- `Career Opportunities`
- `Student Achievements`
- `Placement Facts ...`
- year/trimester headings

This is intentionally pragmatic rather than a generalized document-layout engine.

### Block types

The chunker classifies blocks approximately as:

```text
heading
paragraph
bullet_list
 table
mixed
```

The goal is not perfect semantic document parsing. The goal is to keep related content together and preserve enough section information to improve retrieval.

---

## 7. Embedding generation

Implemented in:

```text
app/services/embedding_service.py
app/llm/openai_client.py
```

### Model configuration

The project uses:

```text
text-embedding-3-large
3072 dimensions
```

The service embeds **`embedding_text`**, not only the raw `content`.

This is important because the embedding input carries section context.

### Batch size

```text
EMBED_BATCH_SIZE = 64
```

Chunks are embedded in batches rather than one API request per chunk.

The same batching pattern is used for query embeddings.

### Dimension validation

After every embedding call, the OpenAI adapter verifies:

1. number of vectors == number of inputs;
2. each vector length == configured embedding dimensions.

This catches configuration/index mismatches early.

A MongoDB vector index configured for 3072 dimensions cannot safely consume an embedding with a different dimension count.

---

## 8. MongoDB Atlas storage model

Implemented in `app/db/mongodb.py` and `app/db/chunk_repository.py`.

A stored chunk is conceptually shaped like:

```json
{
  "_id": "<chunk_id>",
  "chunk_id": "<chunk_id>",
  "document_id": "<document_id>",
  "source_filename": "PGDM_Healthcare_Brochure_final_Open_new_addition.pdf",
  "page_number": 7,
  "section_path": ["Faculties of Healthcare Program"],
  "section_text": "Faculties of Healthcare Program",
  "block_type": "mixed",
  "content": "...",
  "embedding_text": "...",
  "token_count": 100,
  "embedding": [3072 floating-point values],
  "embedding_model": "text-embedding-3-large",
  "embedding_dimensions": 3072,
  "created_at": "...",
  "updated_at": "..."
}
```

### Upsert strategy

`chunk_repository.py` uses `ReplaceOne(..., upsert=True)` with the deterministic chunk ID.

This makes repeated ingestion of the same chunk idempotent at the document level rather than blindly inserting duplicates.

---

## 9. MongoDB indexing — regular indexes vs Search indexes

This distinction caused real operational confusion during development and is important to understand.

### Regular indexes

The repository creates normal indexes such as:

```text
chunk_id   unique
 document_id
```

These are ordinary MongoDB B-tree indexes.

### Atlas Search / Vector Search indexes

The RAG pipeline requires two additional indexes.

#### Vector index

Conceptually:

```json
{
  "fields": [
    {
      "type": "vector",
      "path": "embedding",
      "numDimensions": 3072,
      "similarity": "cosine"
    },
    {"type": "filter", "path": "document_id"},
    {"type": "filter", "path": "page_number"}
  ]
}
```

#### Text index

Conceptually:

```json
{
  "mappings": {
    "dynamic": false,
    "fields": {
      "content": {"type": "string"},
      "section_text": {"type": "string"},
      "source_filename": {"type": "string"}
    }
  }
}
```

### Why two indexes?

Vector search is good at semantic similarity.

Keyword search is better at:

- exact names;
- years;
- numeric labels;
- specific terminology;
- phrases that OCR may distort semantically but still expose lexically.

The hybrid strategy uses both.

### MongoDB Compass

Normal indexes appear in the ordinary **Indexes** view.

Atlas Search/Vector Search indexes appear in the separate **Search Indexes** view.

Therefore, seeing only a few normal indexes in Compass does not prove the vector index exists. The correct diagnostic is the Search Indexes view or the provided diagnostic script.

---

## 10. Index management scripts

### `scripts/create_indexes.py`

This script:

1. connects to the configured MongoDB database;
2. lists current Search indexes;
3. determines which configured indexes are missing;
4. creates only missing indexes;
5. polls until both configured indexes report `READY`;
6. fails clearly if index creation enters a failure state or times out.

The creation logic is intentionally idempotent.

### `scripts/check_mongodb.py`

This script reports:

- MongoDB connectivity;
- active database;
- chunks collection name;
- whether the chunks collection exists;
- number of stored chunk documents;
- regular MongoDB indexes;
- Atlas Search/Vector Search indexes and their statuses.

This is the first diagnostic command to run when `/search` returns no results.

```bash
uv run python scripts/check_mongodb.py
```

### Async PyMongo detail

The project uses PyMongo's async client.

For cursor-producing methods, the correct pattern is:

```python
cursor = await collection.aggregate(pipeline)
rows = await cursor.to_list(length=limit)
```

Similarly:

```python
cursor = await collection.list_search_indexes()
indexes = await cursor.to_list(length=None)
```

Calling `.to_list()` on the coroutine returned by `aggregate()` or `list_search_indexes()` is incorrect and caused the runtime error encountered during development:

```text
'coroutine' object has no attribute 'to_list'
```

The final repository contains tests specifically covering these await-order cases.

---

## 11. Hybrid retrieval

Implemented in:

```text
app/services/retrieval.py
app/db/search_pipelines.py
app/db/chunk_repository.py
```

### Input

`SearchRequest` can contain:

- original query;
- up to four alternate queries;
- up to twelve keywords;
- `top_k`;
- document/page/section filters.

### Retrieval process

For each unique query:

1. generate an embedding;
2. perform vector search;
3. perform keyword search.

These operations are run concurrently using `asyncio.gather()` where appropriate.

Query embeddings are batched to avoid one OpenAI request per query reformulation.

### Vector retrieval

The Atlas Vector Search stage uses:

```text
index      = configured vector index
path       = embedding
numCandidates = max(limit * 20, 100)
limit      = requested candidate limit
similarity = cosine
```

The search result receives a `vectorSearchScore`.

### Keyword retrieval

The Atlas Search stage queries:

```text
content
section_text
source_filename
```

and exposes the Atlas `searchScore`.

### Why raw scores are not directly combined

Vector similarity scores and lexical search scores have different meanings and scales.

Adding them directly would make the ranking sensitive to whichever search engine happens to emit larger numeric values.

Instead, the project uses **Reciprocal Rank Fusion**.

---

## 12. Reciprocal Rank Fusion

Implemented in `app/services/retrieval.py`.

The RRF constant is:

```text
RRF_K = 60
```

The contribution of a hit at rank `r` is approximately:

```text
1 / (RRF_K + r)
```

A chunk appearing near the top in multiple modalities accumulates score.

Example:

```text
vector ranking:
  A
  B
  C

keyword ranking:
  B
  C
  D
```

`B` receives a contribution from both lists and can outrank a chunk that appears near the top of only one list.

This is useful for the brochure because a faculty name can be semantically relevant while the exact phrase `Healthcare Program` is also a strong lexical signal.

### Deduplication

Fusion keys on `chunk_id` so the same chunk retrieved through multiple queries/modalities is not duplicated in the final result set.

---

## 13. Search endpoint

### `POST /search`

Request example:

```json
{
  "query": "Who are the faculty members of the Healthcare Program?",
  "alternate_queries": ["Healthcare Program faculty members"],
  "keywords": ["faculty", "Healthcare Program"],
  "top_k": 8
}
```

Response shape:

```json
{
  "query": "...",
  "queries_used": ["..."],
  "results": [
    {
      "chunk_id": "...",
      "page_number": 7,
      "section_path": ["Faculties of Healthcare Program"],
      "content": "...",
      "fused_score": 0.01,
      "vector_score": 0.8,
      "keyword_score": 7.2
    }
  ]
}
```

### Diagnostic interpretation

If `/search` returns:

```json
{"results": []}
```

then the problem is normally upstream of answer generation:

- no embedded chunks exist;
- wrong database/collection;
- Search index does not exist or is not `READY`;
- configured index name does not match the actual Atlas index;
- search pipeline is pointed at the wrong field;
- filters exclude everything.

The project deliberately does **not** weaken hallucination validation to compensate for empty retrieval.

---

## 14. Conversation memory

Implemented in `app/services/conversation.py`.

Conversation storage is intentionally simple:

```text
conversation document
  ├── _id = UUID string
  ├── created_at
  ├── updated_at
  └── messages[]
       ├── role
       ├── content
       └── created_at
```

The repository returns the most recent messages up to a small limit.

### Why conversation history is used before retrieval

A follow-up like:

```text
What about their healthcare program?
```

is ambiguous without the previous turn.

The history is passed to the search planner so it can resolve references such as:

- “that program”;
- “it”;
- “their placement figures”.

The retrieved evidence still comes only from MongoDB; conversation history is used to clarify the current query, not to invent factual evidence.

---

## 15. LLM provider adapter

Implemented in `app/llm/openai_client.py`.

There are two primary LLM operations.

### 15.1 Search planning

The LLM is instructed:

- never to answer the user directly;
- always to call `search_knowledge_base`;
- to create focused queries for different evidence needs;
- to reformulate ambiguous wording into document terminology;
- to include exact names, dates, years, or numeric labels as keywords.

The function tool schema is generated from `SearchToolArgs` and made strict:

```text
queries
keywords
top_k
filters
```

The code expects **exactly one function call**.

If the model does not return exactly one expected function call, planning fails and is retried within the search-planning retry boundary.

### 15.2 Final grounded answer

The final model is called with structured parsing into `GroundedAnswer`.

The final instructions say:

- answer only from retrieved evidence;
- do not use outside knowledge;
- every factual claim must be supported;
- use exact numbers/labels from evidence;
- do not “correct” OCR;
- citations must reference supplied chunks only;
- for insufficient evidence, set `insufficient_evidence=true`;
- for supported answers, set `grounded=true` and cite evidence.

The use of a typed structured output makes the downstream validation deterministic.

---

## 16. Multi-hop and query reformulation

The assessment asks for multi-hop reasoning and query reformulation.

The final project implements this as **planner-driven multi-query evidence retrieval** rather than a general autonomous agent.

### Example

Question:

```text
Share the placement facts for the Healthcare Program.
```

The planner can generate focused queries such as:

```text
Healthcare Program placement facts
Placement Facts 2021-2023
Placement Facts 2020-2022
```

The retriever then executes the unique queries together, runs vector + keyword retrieval, and fuses the evidence.

### Why this is useful

A single query may retrieve the placement heading but miss one year's figures.

Splitting the question into evidence-focused queries increases the chance that all relevant fact clusters are retrieved before the final answer is generated.

### What this is not

It is not a fully autonomous “search → inspect missing evidence → search again” agent loop.

The implementation was intentionally kept simpler:

```text
LLM plan once
→ multi-query retrieval
→ fused evidence
→ final LLM
```

That is sufficient for the assessment and keeps latency/cost bounded.

---

## 17. RAG orchestration

Implemented in `app/services/rag.py`.

The main `RAGService.answer()` sequence is:

```text
1. load conversation history
2. LLM search planning
3. normalise/de-duplicate search plan
4. hybrid retrieval
5. if no retrieval → safe abstention
6. format context with chunk IDs/pages/sections
7. final grounded LLM generation
8. validate citations/numbers/evidence overlap
9. retry final generation when validation fails
10. append user + assistant messages to conversation
11. return structured response
```

### Context formatting

Each retrieved chunk is given an explicit identifier, for example:

```text
[CHUNK 1: <chunk_id> | page 7 | section: Faculties of Healthcare Program]
Dr. Anjali Kumar ...
```

The model can therefore cite a concrete chunk ID and page number.

---

## 18. Hallucination minimisation and fail-closed behavior

This is one of the most important design decisions in the repository.

The service does not assume that “the LLM answered” means “the answer is acceptable.”

### 18.1 Retrieval-empty guard

If the retriever returns no chunks:

```python
if not retrieved:
    answer = SAFE_ABSTENTION
```

Generation is skipped entirely.

The abstention is deterministic:

```text
I do not have enough evidence in the uploaded documents to answer that reliably.
```

### 18.2 Citation validation

Every citation must satisfy:

```text
citation.chunk_id exists in retrieved results
AND
citation.page_number equals the retrieved page for that chunk
```

A citation to a made-up chunk or mismatched page causes validation failure.

### 18.3 Grounded-answer requirement

If:

```text
grounded = true
```

then at least one citation is required.

The model also cannot claim:

```text
grounded = true
AND
insufficient_evidence = true
```

simultaneously.

### 18.4 Numeric validation

The answer is scanned for numeric literals.

All numeric values must also appear in the cited evidence.

This is especially important for questions such as:

```text
What were the placement figures?
```

because a model could otherwise produce a plausible-looking salary value that was not in the brochure.

### 18.5 Lightweight lexical overlap

The answer and cited evidence are compared using normalized lexical tokens.

A grounded answer must have enough overlap to indicate that it is actually based on retrieved content.

This is intentionally a **lightweight deterministic check**, not a full semantic entailment model.

### 18.6 Bounded retries

If the final answer fails validation, the model gets another attempt.

Maximum:

```text
3 final-generation attempts
```

After that:

```text
SAFE_ABSTENTION
```

is returned.

This is preferable to returning a response that the program has already identified as unsupported.

---

## 19. Retry policy and why it is narrow

There are only two retry boundaries:

### Search planner

```text
max 3 attempts
```

A planner failure can be transient or malformed structured output.

### Final generator

```text
max 3 attempts
```

A final response can be structurally valid but fail the deterministic grounding checks.

### What is intentionally not retried

The system does not wrap all external operations in a generic retry decorator.

MongoDB, OCR, embeddings, and retrieval failures are surfaced.

Why?

Because retries can hide root causes and multiply costs/latency. It is better for this assessment pipeline to have predictable failure boundaries.

---

## 20. Conversation API behavior

### Request

```json
{
  "question": "Who are the faculty members of the Healthcare Program?",
  "conversation_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
  "top_k": 8
}
```

### Response

Typical successful shape:

```json
{
  "conversation_id": "...",
  "answer": "...",
  "citations": [
    {
      "chunk_id": "...",
      "page_number": 7
    }
  ],
  "grounded": true,
  "insufficient_evidence": false
}
```

A deliberate abstention looks like:

```json
{
  "conversation_id": "...",
  "answer": "I do not have enough evidence in the uploaded documents to answer that reliably.",
  "citations": [],
  "grounded": false,
  "insufficient_evidence": true
}
```

That is a **successful RAG outcome** when the evidence is genuinely insufficient.

---

## 21. Endpoint-by-endpoint operational guide

### `GET /health`

Use:

```bash
curl http://127.0.0.1:8000/health
```

Expected:

```json
{"status":"ok"}
```

This only verifies the API process, not MongoDB/OpenAI readiness.

### `POST /ingest/preview`

Swagger UI is the easiest way to upload the PDF.

Expected conceptual response:

```json
{
  "document_id": "...",
  "source_filename": "...pdf",
  "pages": 16,
  "pages_with_native_text": 0,
  "pages_with_ocr": 16,
  "chunks": 100,
  "sample_chunks": [ ... ]
}
```

The exact chunk count can vary if chunking rules or document content changes.

### `POST /ingest`

Use the same PDF upload.

This endpoint requires:

- Tesseract available when OCR is needed;
- valid OpenAI API key;
- valid MongoDB URI;
- reachable MongoDB instance.

### `POST /search`

Use this first when debugging `/chat`.

If `/search` returns no results, do not debug the final LLM yet.

### `POST /chat`

Debug this only after retrieval is proven.

---

## 22. Recommended operational sequence

```bash
# 1. Environment
uv sync

# 2. OCR dependency
which tesseract
tesseract --version

# 3. MongoDB diagnostics
uv run python scripts/check_mongodb.py

# 4. Create Atlas Search/Vector Search indexes
uv run python scripts/create_indexes.py

# 5. Verify again
uv run python scripts/check_mongodb.py

# 6. Preview the document
uv run python scripts/preview_pdf.py docs/PGDM_Healthcare_Brochure_final_Open_new_addition.pdf

# 7. Start API
uv run uvicorn app.main:app --reload

# 8. Test retrieval directly
POST /search

# 9. Test generation
POST /chat

# 10. Run evaluation
uv run python scripts/evaluate.py --base-url http://127.0.0.1:8000 --repeats 2
```

---

## 23. Debugging guide from the actual issues encountered

### Problem: Hatchling cannot determine package files

Error pattern:

```text
ValueError: Unable to determine which files to ship inside the wheel
```

Cause:

The project name `rag-fastapi-mongodb` does not correspond to a Python package directory named `rag_fastapi_mongodb`.

Fix:

```toml
[tool.hatch.build.targets.wheel]
packages = ["app"]
```

### Problem: Tesseract not installed

Error:

```text
tesseract is not installed or it's not in your PATH
```

Cause:

`pytesseract` exists, but the Tesseract binary does not.

Fix:

```bash
sudo apt update
sudo apt install -y tesseract-ocr
```

### Problem: async tests not supported

Error pattern:

```text
async def functions are not natively supported
Unknown pytest.mark.asyncio
```

Cause:

`pytest-asyncio` was missing from dev dependencies.

Fix:

```toml
dev = ["pytest>=8.0", "pytest-asyncio>=1.4.0", "ruff>=0.8"]
```

### Problem: chunk size test exceeded 520 tokens

Cause:

The original splitting logic did not guarantee the public token-count function and the splitter boundary remained consistent in all environments.

Fix:

The splitter now performs a defensive post-check and backs off until `token_count(piece) <= 520`.

### Problem: `/chat` returns 502 with `coroutine` / `to_list`

Error:

```text
'coroutine' object has no attribute 'to_list'
```

Cause:

An async PyMongo cursor-producing operation was not awaited before `to_list()`.

Correct pattern:

```python
cursor = await collection.aggregate(pipeline)
rows = await cursor.to_list(length=limit)
```

### Problem: `/search` returns empty results

Most likely causes:

1. no chunks in the active database;
2. wrong `MONGODB_DB` setting;
3. vector/search index missing;
4. search index not `READY`;
5. index name mismatch;
6. embedding data missing;
7. retrieval filters excluding all documents.

First command:

```bash
uv run python scripts/check_mongodb.py
```

Then verify Atlas Search indexes in the Search Indexes view.

### Problem: `/chat` returns a clean 200 with `insufficient_evidence=true`

This is not necessarily an error.

Two possibilities:

- retrieval returned no chunks; or
- the final generated answer failed grounding validation after the retry budget.

Therefore test `/search` independently before changing prompts or guards.

---

## 24. Evaluation harness

The evaluation fixture contains three assessment questions.

### Case 1 — Welingkar identity

Question:

```text
What is Welingkar?
```

Expected evidence includes the institute identity and WeSchool/management-education context.

### Case 2 — faculty

Question:

```text
Who are the faculty of the Healthcare Program?
```

Expected evidence is on page 7 and covers six named faculty members.

### Case 3 — placement facts

Question:

```text
Share the placement facts for the Healthcare Program.
```

Expected evidence is on page 11 and covers the 2021–2023 and 2020–2022 placement figures.

### Evaluation metrics

`app/services/evaluation.py` computes:

#### Accuracy

Fraction of expected fact groups detected in the answer.

#### Relevance

Fraction of expected topic terms found in the normalized answer.

#### Citation grounding

Whether the response cites at least one expected page, with a special safe-abstention case.

#### Consistency

For repeated runs, detected fact IDs are compared using pairwise Jaccard similarity.

#### Overall score

Weights:

```text
accuracy             55%
relevance            20%
citation grounding   15%
consistency          10%
```

This weighting intentionally emphasizes factual correctness.

### Important limitation

The evaluator is deterministic and brochure-specific.

It is useful for regression testing, but it does not fully measure semantic correctness, completeness, fluency, or nuanced multi-hop reasoning.

---

## 25. Tests and what they protect

The repository contains **15 tests**.

### Chunking tests

Protect:

- heading awareness;
- long-text splitting;
- 520-token hard ceiling;
- section continuity.

### Embedding tests

Protect:

- mapping vectors back to the correct chunks;
- dimension metadata propagation.

### Search pipeline tests

Protect:

- vector search configuration;
- candidate counts;
- vector filters;
- lexical search score mapping.

### Retrieval tests

Protect:

- RRF ranking behavior;
- hybrid search output size;
- batch query embeddings;
- cross-modality overlap.

### Repository tests

Protect:

- async `aggregate()` await order;
- async `to_list()` use after obtaining the cursor.

### Index tests

Protect:

- vector index path/dimension configuration;
- async `list_search_indexes()` handling.

### RAG tests

Protect:

- bounded retries on the planner;
- successful conversation persistence;
- invalid citation fail-closed behavior;
- unsupported number fail-closed behavior.

### Evaluation tests

Protect:

- expected fact detection;
- citation grounding logic;
- consistency score;
- weighted overall score.

Run:

```bash
uv run pytest -q
```

---

## 26. Data flow for the faculty question

Consider:

```text
Who are the faculty members of the Healthcare Program?
```

### Step 1 — user request

`POST /chat` receives the question.

### Step 2 — history

The conversation repository loads recent messages if `conversation_id` is present.

### Step 3 — planner

The OpenAI planner is instructed to call `search_knowledge_base` and can produce a focused query such as:

```text
Healthcare Program faculty members
```

plus keywords such as:

```text
faculty
Healthcare Program
```

### Step 4 — retrieval

The retriever embeds the query and performs:

- vector search;
- keyword search.

### Step 5 — RRF

The ranked lists are fused by chunk ID.

### Step 6 — context

Page-7 faculty chunks are formatted with chunk IDs and page numbers.

### Step 7 — generation

The final LLM generates a structured `GroundedAnswer`.

### Step 8 — validation

The service checks:

- all citations exist;
- page numbers match;
- answer numbers are supported;
- answer text overlaps cited evidence.

### Step 9 — response

If valid:

```text
grounded=true
insufficient_evidence=false
```

Otherwise, after retries, the service abstains.

---

## 27. Data flow for the placement question

Question:

```text
Share the placement facts for the Healthcare Program.
```

This is a better demonstration of the hybrid/multi-query design because there are multiple fact groups and years.

Expected evidence exists on page 11.

The planner is encouraged to preserve exact labels such as:

```text
2021-2023
2020-2022
Maximum Salary
Median Salary
Average Salary
```

This matters because numeric data is vulnerable to hallucination and OCR distortion.

The final answer is only accepted when all cited numeric values can be found in the cited evidence.

---

## 28. Security and secret handling

Secrets are intentionally kept outside the repository.

The local `.env` file should contain credentials but must not be committed.

The project archive prepared for the assessment excludes credentials.

The MongoDB URI and OpenAI key should never be placed in:

- source code;
- test fixtures;
- README examples;
- `evaluation/results`;
- committed shell history scripts.

Only placeholders belong in `.env.example`.

---

## 29. Why LangChain was not used

The project deliberately avoids LangChain.

For this assessment, the additional abstraction was not necessary because the pipeline needs only a small number of explicit operations:

```text
PDF extraction
chunking
embedding
MongoDB retrieval
RRF
LLM tool call
structured generation
validation
conversation persistence
```

Keeping those functions explicit makes it easier to show exactly:

- where data changes shape;
- where retries happen;
- where evidence is enforced;
- where provider-specific code lives;
- where MongoDB search is called.

This also makes future framework migration optional rather than fundamental.

---

## 30. Future upgrade path

The project is intentionally ready for later upgrades without implementing them now.

### LiteLLM

The planned insertion point is the provider boundary:

```text
app/llm/openai_client.py
```

The rest of the application should continue to depend on service methods such as:

```text
embed()
plan_search()
generate_grounded_answer()
```

rather than directly importing an SDK everywhere.

### Langfuse

Observability could later instrument:

- planning LLM calls;
- embedding calls;
- retrieval latency;
- final LLM calls;
- validation failures;
- token usage;
- evaluation runs.

No Langfuse code is part of the final assessment implementation.

### More advanced multi-hop loop

A future production version could evolve:

```text
planner
→ search
→ inspect evidence gaps
→ second search
→ final answer
```

This repository deliberately stops at:

```text
planner
→ multi-query search
→ fused evidence
→ final answer
```

### Better grounding evaluation

A production evaluation stack could add:

- semantic entailment scoring;
- retrieval recall@k;
- precision@k;
- answer completeness;
- answer faithfulness;
- LLM-as-judge with calibrated rubrics;
- human review samples.

---

## 31. Final assessment status

The final project is an end-to-end basic RAG implementation with these completed capabilities:

- PDF processing with native extraction and OCR fallback;
- structure-aware chunking;
- hard token ceiling;
- OpenAI embedding generation;
- MongoDB Atlas storage;
- vector and lexical Search indexes;
- hybrid retrieval;
- reciprocal rank fusion;
- LLM query planning and reformulation;
- multi-part evidence retrieval;
- conversation continuity;
- structured grounded answer generation;
- citation validation;
- numeric grounding validation;
- evidence overlap validation;
- bounded LLM retries;
- fail-closed abstention;
- automated assessment evaluation;
- focused unit tests for the major correctness boundaries;
- explicit provider boundary for later LiteLLM/Langfuse adoption.

The system is intentionally conservative. Some unsupported or weakly retrieved questions can return a clean “insufficient evidence” response instead of an answer. That is a designed property of the final pipeline, not an accidental feature.

---

## 32. Quick reference

### Start

```bash
uv sync
uv run uvicorn app.main:app --reload
```

### MongoDB diagnostics

```bash
uv run python scripts/check_mongodb.py
```

### Create Search/Vector indexes

```bash
uv run python scripts/create_indexes.py
```

### Preview PDF extraction/chunks

```bash
uv run python scripts/preview_pdf.py docs/PGDM_Healthcare_Brochure_final_Open_new_addition.pdf
```

### Tests

```bash
uv run pytest -q
```

### Evaluation

```bash
uv run python scripts/evaluate.py --base-url http://127.0.0.1:8000 --repeats 2
```

### Swagger

```text
http://127.0.0.1:8000/docs
```

---

## 33. Final note on correctness

The core design intentionally separates three questions that are often incorrectly conflated:

```text
Did retrieval find evidence?
Did the model produce a structured answer?
Did the answer survive grounding validation?
```

A `200` response with `insufficient_evidence=true` can therefore be the correct result even when the endpoint itself is healthy.

Likewise, an empty `/search` response should be debugged at the data/index/retrieval layer before changing prompts or weakening the final grounding gate.

That separation is one of the main reliability characteristics of the final implementation.
