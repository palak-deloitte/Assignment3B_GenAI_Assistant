# Assignment 3B: GenAI Assistant

A document-aware chat assistant. Users log in, upload PDF/TXT files, and chat. A LangGraph agent decides, for every message, whether to answer from the user's documents (RAG, with citations), call a weather API, or reply directly. Conversation memory lasts for the session.

## Architecture

```
Streamlit UI        login · upload · chat · citations
      │  HTTP + token
Django / DRF        auth · history · POST /api/chat
      │
LangGraph agent     router ─┬─ documents → FAISS search → Claude answers with sources
                            ├─ weather   → find city → OpenWeather API → Claude answers
                            └─ general   → Claude answers directly
      │
Claude (LLM)  ·  FAISS vector store  ·  OpenWeather API (external tool)
```

The project is built on the Assignment 2 and 3A backends.

## Features

| Requirement | Implementation |
|---|---|
| Document upload | Streamlit file uploader accepts several PDF/TXT files (up to 5 MB each). Each is uploaded, chunked and embedded, and the UI shows the chunk count. |
| Multi-document indexing | Every user has one FAISS index holding chunks from all their documents. Each chunk carries its `document_id`, and a single answer can cite several documents. |
| Chat interface | Streamlit chat UI. Earlier turns stay visible during the session. |
| Memory | LangGraph `MemorySaver` keyed by `thread_id = "<user id>:<session id>"`. Follow-ups such as "what about its humidity?" are resolved from earlier turns. |
| Agentic flow | A router node with conditional edges to three different branches (documents, weather, general). |
| External tool | OpenWeather current-weather API, called from the weather branch when the user asks about the weather. |
| Source citations | Retrieved chunks are shown under the answer in a "Sources" expander, with document titles. |
| Auth | Login and register screen. The chat is not reachable without a token. Tokens come from the Django backend and expire after 24 hours. |
| Tests | End-to-end tests drive the real graph through `/api/chat` (message, routing, tool call, final answer) with the LLM and weather API faked. |

## How the agent works

1. **Router:** the model reads the recent conversation and replies with one word: `documents`, `weather` or `general`. Unclear answers default to `general`.
2. **Documents branch:** on follow-up questions, the question is first rewritten as a standalone question using the conversation. The user's index is then searched, and the model answers using only the retrieved chunks, citing them as [1], [2]. If nothing is found, the user is told so.
3. **Weather branch:** the model extracts the city from the whole conversation (so follow-ups work), the OpenWeather API is called, and the model writes the answer from the returned data only. An unknown city, a missing city, or an API failure each get a clear message, never a crash.
4. **General branch:** a direct reply.

Every prompt uses a fixed system message and a placeholder for user text, so user input is never sent to the model on its own. Retrieved document text is placed inside `<context>` tags, with an instruction to treat it as reference material and never as instructions.

## API

All endpoints except register and login need `Authorization: Token <token>`.

| Method | Path | Description |
|---|---|---|
| POST | `/auth/register`, `/auth/login` | Create an account or log in; returns a token |
| POST | `/auth/logout` | Invalidate the token |
| POST | `/documents` | Upload a .txt/.pdf file or send JSON `title` + `content` |
| GET | `/documents`, `/documents/{id}` | List or retrieve your documents (paginated list) |
| DELETE | `/documents/{id}` | Delete a document and its embeddings |
| POST | `/api/ingest-doc` | Chunk and embed a document |
| POST | `/api/rag-query` | One-shot RAG question; returns answer and sources |
| POST | `/api/ask-ai` | One-shot question through a prompt template |
| **POST** | **`/api/chat`** | **Agent turn. Body `{"message": "...", "session_id": "..."}`. Returns `{"answer", "sources", "route", "tool"}`** |
| GET | `/api/query-history` | Your AI history, including chat turns (paginated) |

`/api/chat` status codes: 200 success; 400 empty or too long message (max 2000 characters) or invalid `session_id` (8 to 64 letters, digits, `-` or `_`); 401 missing or expired token; 502 the LLM provider failed; 503 an internal failure such as the vector store.

## Configuration

| Setting | Value |
|---|---|
| LLM | Claude (`claude-haiku-5-5` by default, set with `LLM_MODEL`) through `langchain-anthropic` |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2`, run locally |
| Vector store | FAISS, one index per user in `vector_store/user_<id>/` |
| Chunk size / overlap | 1000 / 200 characters (`RecursiveCharacterTextSplitter`) |
| Chunks retrieved | 4 per query |
| Conversation window | the model sees the last 10 messages |
| Weather | OpenWeather current weather, metric units, 5 s timeout |

## Setup

Requires Python 3.12.

```bash
python -m venv venv
venv\Scripts\activate            # Windows
# source venv/bin/activate       # macOS / Linux
pip install -r requirements.txt
python manage.py migrate
```

Create `.env` (see `.env.example`):

| Variable | Description |
|---|---|
| `ANTHROPIC_API_KEY` | LLM key |
| `OPENWEATHER_API_KEY` | Key from openweathermap.org |
| `LLM_MODEL` | Optional, defaults to `claude-haiku-5-5` |
| `BACKEND_URL` | Optional, used by Streamlit. Defaults to `http://127.0.0.1:8000` |

## Run

Use two terminals from the project folder.

```bash
# Terminal 1: backend
python manage.py runserver

# Terminal 2: UI
streamlit run streamlit_app/app.py
```

Open http://localhost:8501, register, upload documents in the sidebar, and start chatting. Use **New conversation** in the sidebar to reset the agent's memory.

The first document upload or question after the backend starts takes longer, because the embedding model (about 90 MB, downloaded on first use) is loaded into memory once per server process.

## Tests

```bash
python -m pytest -v
```

The tests are fully offline. The LLM is replaced by a scripted fake that recognises each kind of prompt and records the sequence of calls. OpenWeather's HTTP call is faked, and embeddings are replaced by a small keyword model. The real parts are the Django views and authentication, the LangGraph graph with its router and memory, the prompt templates, the weather parsing code, chunking and FAISS (written to a temporary folder).

`api/tests/test_agent.py` includes the end-to-end tests (`test_weather_end_to_end`, `test_documents_end_to_end`) as well as tests for memory, routing, per-user isolation, tool failures, and error codes.

## Project structure

```
config/                project settings
api/
  agent.py             LangGraph agent: router, branches, memory
  weather.py           the OpenWeather tool
  llm.py               prompt templates and the Claude client
  rag.py               chunking, embeddings, FAISS storage and search
  views.py             API views (including /api/chat)
  models.py, serializers.py, authentication.py, uploads.py, ...
  tests/               automated tests
streamlit_app/
  app.py               the Streamlit UI
  api_client.py        HTTP calls to the backend
```

## Limitations

- **Memory is held in the backend process.** It resets when the Django server restarts, which the dev server does whenever a file changes. Persistent memory would need a database-backed checkpointer.
- **The UI shows only the current conversation.** A browser refresh starts a new Streamlit session and requires a new login. Past turns are saved and available from `GET /api/query-history`, but not browsable in the UI.
- Document search returns the nearest chunks even when they are only loosely related, so the cited sources can include chunks that are not very relevant. A similarity-score cutoff would be the fix.
- Ingestion happens inside the request, so large documents take a while. A production system would use a background worker.
- Scanned PDFs without a text layer are rejected.
- The FAISS index is a local file. A multi-server deployment would need a hosted vector database.
- The Claude API quota is shared, so heavy use can return 502 when requests are rate-limited.