# Assignment 3A: AI Backend (Django + LLM)

A Django REST Framework backend with token authentication, document management, and AI endpoints for prompt-template Q&A, document ingestion (chunking and embeddings), and retrieval-based question answering (RAG) with source chunks.

This project extends the Assignment 2 backend.

## Endpoints

### AI endpoints

| Method | Path | Description |
|---|---|---|
| POST | `/api/ask-ai` | Body `{"prompt": "..."}`. The prompt is inserted into a fixed prompt template and sent to the LLM. Returns `{"response": "..."}` |
| POST | `/api/ingest-doc` | Body `{"document_id": 1}`. Chunks the document and stores embeddings. Returns `{"document_id", "status", "chunk_count"}` |
| POST | `/api/rag-query` | Body `{"query": "..."}`. Retrieves relevant chunks and generates an answer. Returns `{"answer": "...", "sources": [{"document_id", "chunk_text"}]}` |
| GET | `/api/query-history` | Your AI query history (paginated) |

### Account and document endpoints (from Assignment 2)

| Method | Path | Description |
|---|---|---|
| POST | `/auth/register`, `/auth/login` | Create an account or log in; returns a token (open endpoints) |
| POST | `/auth/logout` | Invalidate the current token |
| POST | `/documents` | Upload a .txt/.pdf file or send JSON `title` + `content` |
| GET | `/documents` | List your documents (paginated) |
| GET | `/documents/{id}` | Retrieve one of your documents |
| DELETE | `/documents/{id}` | Delete a document **and its embeddings** |
| POST | `/query` | Record a plain query in your history |
| GET | `/queries` | List all your recorded queries (paginated) |

All endpoints except register and login require the header `Authorization: Token <token>`. Tokens expire after 24 hours.

### Status codes (AI endpoints)

| Endpoint | Code | When |
|---|---|---|
| `/api/ask-ai` | 200 | Success |
| | 400 | `prompt` missing, empty, or longer than 4000 characters |
| | 401 | Token missing, invalid, or expired |
| | 502 | The LLM provider failed (including rate limits) |
| `/api/ingest-doc` | 200 | Document ingested |
| | 400 | Invalid `document_id`, or the document has no text |
| | 401 | Token missing, invalid, or expired |
| | 404 | Document does not exist or is not yours |
| | 503 | Indexing failed internally |
| `/api/rag-query` | 200 | Success (also returned when you have no ingested documents, with an explanatory answer and empty `sources`) |
| | 400 | `query` missing, empty, or longer than 2000 characters |
| | 401 | Token missing, invalid, or expired |
| | 502 | The LLM provider failed |
| | 503 | Searching the vector store failed internally |
| `/api/query-history` | 200 | Paginated results |
| | 401 | Token missing, invalid, or expired |
| | 404 | `page` out of range |

Pagination is `page` and `page_size` (default 10, maximum 100) and returns `count`, `next`, `previous`, `results`, newest first. History entries include `query_text`, `result_summary` (the AI's answer), `kind` (`ask_ai` or `rag`), and `created_at`.

## AI design

| Setting | Value |
|---|---|
| **LLM** | Claude (`claude-haiku-5-5` by default), through `langchain-anthropic`. Configurable with `LLM_MODEL`. |
| **Embedding model** | `sentence-transformers/all-MiniLM-L6-v2`, run locally (384-dimensional vectors) |
| **Vector store** | **FAISS** (`faiss-cpu`) through LangChain, used for all retrieval |
| **Chunk size** | **1000 characters** |
| **Chunk overlap** | **200 characters** |
| **Chunking method** | `RecursiveCharacterTextSplitter` (prefers paragraph, then sentence, then word boundaries). Text is cleaned first, so soft line breaks (common in PDFs) become spaces. |
| **Chunks retrieved per query** | 4 (`RAG_TOP_K`) |
| **LLM limits** | 30 s timeout, 1024 max output tokens |

These values are defined in `config/settings.py`.

- **Why local embeddings:** the LLM provider does not offer an embeddings API, and a local model needs no extra key, has no rate limits, and keeps document text on the machine.
- **Per-user indexes:** each user has their own FAISS index in `vector_store/user_<id>/`. A user can only ever search their own chunks, so documents cannot leak between users. Each chunk stores its `document_id` in metadata.
- **Re-ingesting** a document replaces its old chunks. **Deleting** a document removes its chunks.
- **Prompt templates:** user input is never sent to the model on its own. `/api/ask-ai` and `/api/rag-query` use LangChain `ChatPromptTemplate`s with fixed system instructions, and the user text fills a placeholder in the human message.
- **RAG prompt:** the model is told to answer only from the numbered context passages, to say so when the answer is not there, and to treat the context as reference material only. This reduces the effect of instructions hidden inside uploaded documents.
- **History:** AI calls are recorded in `QueryLog` with a `kind` field. Only successful calls are recorded.

## Setup

Requires Python 3.12.

```bash
python -m venv venv
venv\Scripts\activate            # Windows
# source venv/bin/activate       # macOS / Linux
pip install -r requirements.txt
python manage.py migrate
python manage.py createsuperuser   # optional, for /admin/
python manage.py runserver
```

Create a `.env` file (see `.env.example`):

| Variable | Description |
|---|---|
| `ANTHROPIC_API_KEY` | API key for the LLM |
| `LLM_MODEL` | Optional. Defaults to `claude-haiku-5-5` |
| `HF_HUB_OFFLINE` | Optional. Set to `1` after the first run to stop checking Hugging Face for model updates |

**First run:** the first request that needs embeddings downloads the embedding model (about 90 MB) and loads it into memory, which takes some time. The model is loaded once per server process, and later requests are fast. Restarting the dev server (it restarts itself when files change) loads the model again.

Typical flow: register, log in, `POST /documents`, `POST /api/ingest-doc`, then `POST /api/rag-query`.

## Tests

```bash
python -m pytest -v
```

The tests are fully offline. The LLM is replaced by a fake that records the messages it receives (so tests can check what the model was sent), embeddings are replaced by a small keyword-based fake, and the vector store is written to a temporary folder. The tests cover chunking, indexing, re-indexing and deletion, retrieval relevance and multi-document retrieval, per-user isolation, prompt-template usage, error mapping (400, 401, 404, 502, 503), and pagination.

## Project structure

```
config/              project settings and root URLs
api/
  models.py          Document, QueryLog (with kind), UserSession
  authentication.py  token authentication backed by UserSession
  views.py           API views
  serializers.py     request and response validation
  llm.py             prompt templates and the Claude client
  rag.py             chunking, embeddings, FAISS storage and search
  uploads.py         file validation and text extraction
  pagination.py, permissions.py
  tests/             automated tests
```

## Limitations

- Ingestion runs inside the request, so large documents take a while. A production system would use a background worker.
- Similarity search always returns the nearest chunks, even when none are truly relevant. The prompt makes the model say when the context does not contain the answer, but a similarity-score threshold would be a stricter fix.
- Scanned PDFs without a text layer are rejected, since there is no text to extract.
- FAISS indexes are local files, and metadata is stored with pickle, which is only safe because this application writes every file it loads. A deployment with several servers would need a hosted vector database.
- Requests share the Claude API quota, so heavy use can return 502 when the provider rate-limits calls.