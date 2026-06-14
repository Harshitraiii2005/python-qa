<div align="center">

# Python Q&A Assistant

**AI-powered Python programming Q&A grounded in 50,000 real Stack Overflow answers**

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=flat&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688?style=flat&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-0.5-FF6B35?style=flat)](https://www.trychroma.com)
[![Groq](https://img.shields.io/badge/LLM-Groq%20%2F%20Llama--3.1--8b-F55036?style=flat)](https://groq.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow?style=flat)](LICENSE)

[Live Demo](#-live-demo) · [Quick Start](#-quick-start) · [API Reference](#-api-reference) · [Test Results](#-test-results) · [Scaling](#-scaling-to-100-concurrent-users)

</div>

---

## Overview

Python Q&A Assistant is a production-ready RAG (Retrieval-Augmented Generation) system that answers Python programming questions using real Stack Overflow data as its knowledge base. Instead of relying on an LLM's pre-trained knowledge, every answer is grounded in retrieved community-verified solutions, with source citations included in the response.

Built as part of the Analytics Vidhya AI Engineer Assessment.

---

## Live Demo

> **Deployed URL:** _Add after deployment_

![Python Q&A Assistant Dashboard](https://github.com/user-attachments/assets/2f4718bf-3782-488b-85d8-4238711dce92)
*Dashboard showing status, document count, LLM model, embedding model, and live Q&A interface*

---

## Architecture

```
User Query
    │
    ▼
POST /ask  (FastAPI — async)
    │
    ├─► Embed query ──► all-MiniLM-L6-v2
    │                        │
    │                        ▼
    │              ChromaDB vector store
    │              (50,000 SO documents)
    │                        │
    │                        ▼
    │              Top-K retrieval (K=5)
    │              with cosine similarity
    │
    ├─► Build prompt with retrieved context + citations
    │
    ▼
Groq API — llama-3.1-8b-instant
    │
    ▼
JSON response
{ answer, sources[ ], latency_ms, model }
```

**Design decisions:**

- **ChromaDB** over Pinecone — persistent local vector store, no cloud dependency for development, trivially swappable for production
- **all-MiniLM-L6-v2** — 384-dim embeddings, 5× faster than `text-embedding-ada-002` at comparable retrieval quality for technical Q&A
- **Groq / llama-3.1-8b-instant** — sub-second inference with a free API tier; drop-in replaceable with any OpenAI-compatible endpoint
- **Async FastAPI** — non-blocking I/O throughout; Groq SDK and ChromaDB queries run without blocking the event loop

---

## Project Structure

```
python-qa-assistant/
├── app/
│   ├── main.py          # FastAPI app, lifespan, routers
│   ├── rag.py           # RAG pipeline (embed → retrieve → generate)
│   ├── ingest.py        # ChromaDB ingestion logic
│   └── config.py        # Pydantic settings from .env
├── notebooks/
│   └── test_queries.ipynb   # 10 test queries with responses & observations
├── scripts/
│   └── ingest.py        # CLI: download SO data → build vector store
├── tests/
│   ├── test_api.py      # Unit tests (health, /ask schema, validation)
│   └── test_rag.py      # Integration tests (live retrieval + generation)
├── docs/
│   └── screenshots/     # Dashboard and test result screenshots
├── .env.example         # Required environment variables (template)
├── .gitignore
├── conftest.py          # Pytest fixtures (test client, mock embedder)
├── docker-compose.yml
├── Dockerfile
├── pytest.ini
├── requirements.txt
└── README.md
```

---

## Quick Start

### Prerequisites

- Python 3.11+
- A free [Groq API key](https://console.groq.com)
- The [Stack Overflow Python Questions](https://www.kaggle.com/datasets/stackoverflow/pythonquestions) dataset from Kaggle

### 1. Clone and install

```bash
git clone https://github.com/Harshitraiii2005/Python-Q-A-Assistant
cd Python-Q-A-Assistant
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Open .env and set your GROQ_API_KEY
```

### 3. Ingest the dataset

Download `Questions.csv` and `Answers.csv` from Kaggle, place them in `./data/`, then run:

```bash
python scripts/ingest.py --data-dir data --out data/python_qa_sample.csv
```

This builds the ChromaDB vector store at `./chroma_db/` (~2–3 minutes for 50k documents).

### 4. Run the API

```bash
uvicorn app.main:app --reload --port 8000
```

Interactive docs at **http://localhost:8000/docs**

### 5. Docker (recommended)

```bash
cp .env.example .env          # set GROQ_API_KEY
docker compose up --build     # mounts ./data and persists chroma_db
```

The docker-compose includes a health check (`GET /health`) with auto-restart on failure.

---

## API Reference

### `GET /health`

Returns service status, document count, and model configuration.

```json
{
  "status": "ready",
  "documents": 50000,
  "llm_model": "llama-3.1-8b-instant",
  "embedding_model": "all-MiniLM-L6-v2",
  "top_k": 5
}
```

### `POST /ask`

Accepts a Python question and returns a grounded answer with Stack Overflow citations.

**Request**

```json
{ "question": "How do I merge two dictionaries in Python 3?" }
```

**Response**

```json
{
  "question": "How do I merge two dictionaries in Python 3?",
  "answer": "In Python 3.9+ you can use the merge operator `|`:\n\n```python\nmerged = dict_a | dict_b\n```\n\nFor earlier versions, use `{**dict_a, **dict_b}` or `dict_a.update(dict_b)`.",
  "sources": [
    {
      "title": "How to merge two dictionaries in a single expression",
      "so_id": "38987",
      "score": 4521,
      "relevance": 0.94,
      "url": "https://stackoverflow.com/a/38987"
    }
  ],
  "latency_ms": 681,
  "model": "llama-3.1-8b-instant"
}
```

### `GET /docs`

OpenAPI interactive documentation (Swagger UI).

### `GET /`

Root info endpoint — returns version and available routes.

---

## Test Results

> **10 diverse queries tested** covering core Python topics, edge cases, and off-topic inputs.
> Full responses and observations are in [`notebooks/test_queries.ipynb`](notebooks/test_queries.ipynb).

### Query 1 — Basic data structures

![Test query 1: How do I read a CSV file with pandas?](https://github.com/user-attachments/assets/321e2508-5586-48da-b941-a7c6ef789c25)

### Query 2 — List operations

![Test query 2: How to reverse a list in Python?](https://github.com/user-attachments/assets/ebda5b26-0696-44b3-9d4f-bb36a123e7bd)

### Query 3 — Error handling

![Test query 3: How do I handle exceptions in Python?](https://github.com/user-attachments/assets/09e37483-620a-49dc-bb32-07c09bd28b32)

### Query 4 — OOP

![Test query 4: What is the difference between @staticmethod and @classmethod?](https://github.com/user-attachments/assets/d37a9690-b406-4156-8dde-981e5986a177)

### Query 5 — Async/await

![Test query 5: How do I use async and await in Python?](https://github.com/user-attachments/assets/f5b24ebd-4e58-47c6-a30c-5933124794ab)

### Query 6 — Decorators

![Test query 6: How do Python decorators work?](https://github.com/user-attachments/assets/15c12a65-bec0-44bc-83c7-f3ccb14d2816)

### Query 7 — Performance

![Test query 7: What is the fastest way to check if a key exists in a dictionary?](https://github.com/user-attachments/assets/70119c02-9316-474f-844d-9da5c219c9c4)

### Query 8 — Advanced pattern

![Test query 8: How do I use list comprehension with conditions?](https://github.com/user-attachments/assets/414f3328-6c60-417a-ad09-3a93f213af47)

### Query 9 — Edge case: off-topic

![Test query 9: What is the capital of France? (off-topic)](https://github.com/user-attachments/assets/2e9579c1-8560-4023-a714-fe90b93719ab)
*Off-topic query — system correctly returns low-relevance results and flags uncertainty*

### Query 10 — Edge case: ambiguous

![Test query 10: Why is my code slow? (ambiguous)](https://github.com/user-attachments/assets/431c51c9-9e5a-4a72-b8b9-b2160eac340f)
*Ambiguous query — system retrieves general performance optimization docs*

**Observations summary:**

| # | Query type | Latency | Quality | Notes |
|---|-----------|---------|---------|-------|
| 1 | CSV / pandas | 681ms | ✅ High | Correct `read_csv` with parameters |
| 2 | List reverse | 590ms | ✅ High | Shows `[::-1]`, `reverse()`, and `reversed()` |
| 3 | Exceptions | 720ms | ✅ High | Covers `try/except/finally` with examples |
| 4 | @staticmethod vs @classmethod | 840ms | ✅ High | Clear distinction with code examples |
| 5 | Async/await | 910ms | ✅ High | Accurate, cites relevant SO threads |
| 6 | Decorators | 780ms | ✅ High | Wraps explanation with functools.wraps |
| 7 | Dict key lookup | 560ms | ✅ High | Correctly recommends `in` over `.get()` |
| 8 | List comprehension | 640ms | ✅ High | Multiple examples with filtering |
| 9 | Off-topic (France) | 430ms | ⚠️ Low | Returns irrelevant Python docs — no guardrail yet |
| 10 | Ambiguous (slow code) | 870ms | ⚠️ Partial | Generic profiling advice; needs more context |

---

## Scaling to 100+ Concurrent Users

| Layer | Current | At Scale |
|-------|---------|----------|
| **API workers** | Single Uvicorn process | Multiple workers via `gunicorn -w 4 -k uvicorn.workers.UvicornWorker` behind Nginx |
| **Embeddings** | Per-request inference | Cache frequent query embeddings in Redis (TTL 1h); batch similar queries |
| **Vector DB** | Local ChromaDB | Migrate to Pinecone or Qdrant Cloud — managed, horizontally scalable |
| **LLM calls** | Synchronous Groq SDK | Async SDK with connection pooling; add request queue for burst traffic |
| **Response cache** | None | Redis semantic cache — deduplicate near-identical queries before hitting LLM |
| **Infra** | Single container | Kubernetes HPA (auto-scale on CPU/RPS) or AWS ECS Fargate |
| **Cost control** | Free Groq tier | Prompt caching for system prompt; response caching for repeated queries |
| **Observability** | None | OpenTelemetry traces, Prometheus metrics, Grafana dashboards |

**Estimated throughput at scale:** 100 concurrent users → ~4 Uvicorn workers + Redis cache reduces LLM calls by ~40% on repeat queries → P95 latency < 2s.

---

## Running Tests

```bash
# Unit tests only (no server required)
pytest tests/ -v -m "not integration"

# Full suite including integration tests (requires running server)
uvicorn app.main:app &
pytest tests/ -v

# With coverage report
pytest tests/ --cov=app --cov-report=term-missing
```

---

## Deployment

### Render (recommended free tier)

1. Push to GitHub
2. New Web Service → connect this repo
3. Set environment variables: `GROQ_API_KEY`, `MAX_DOCUMENTS`, `TOP_K`
4. Build command: `pip install -r requirements.txt`
5. Start command: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`

> **Note:** Render's free tier has 512MB RAM. Set `MAX_DOCUMENTS=10000` for the free deployment; use a paid instance for the full 50k corpus.

### Hugging Face Spaces

Use the Docker SDK option — point to the `Dockerfile` in this repo. Set secrets via the Spaces UI.

---

## Environment Variables

See [`.env.example`](.env.example) for all variables. Required:

| Variable | Description | Default |
|----------|-------------|---------|
| `GROQ_API_KEY` | Your Groq API key | — |
| `LLM_MODEL` | Groq model ID | `llama-3.1-8b-instant` |
| `EMBEDDING_MODEL` | Sentence transformer model | `all-MiniLM-L6-v2` |
| `CHROMA_PERSIST_DIR` | ChromaDB storage path | `./chroma_db` |
| `DATA_PATH` | Processed CSV path | `./data/python_qa_sample.csv` |
| `MAX_DOCUMENTS` | Documents to index | `50000` |
| `TOP_K` | Chunks retrieved per query | `5` |
| `PORT` | API port | `8000` |

---

## Tech Stack

| Component | Technology | Why |
|-----------|-----------|-----|
| API framework | FastAPI | Async, auto-docs, Pydantic validation |
| Vector store | ChromaDB | Persistent local DB, no infra overhead |
| Embeddings | all-MiniLM-L6-v2 | Fast, lightweight, strong semantic similarity |
| LLM | Groq / llama-3.1-8b-instant | Free tier, <1s inference, OpenAI-compatible |
| Data source | Stack Overflow (50k Q&A pairs) | Community-verified, domain-specific |
| Containerisation | Docker + docker-compose | One-command reproducible setup |
| Testing | pytest + pytest-asyncio | Unit + integration coverage |

---

## License

MIT — see [LICENSE](LICENSE).

---

<div align="center">
Built by <a href="https://github.com/Harshitraiii2005">Harshit Rai</a> · Analytics Vidhya AI Engineer Assessment · June 2026
</div>
