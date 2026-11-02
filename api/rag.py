import re
import threading
from functools import lru_cache
from pathlib import Path

from django.conf import settings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document as LCDocument
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Only one thread may read-modify-write an index file at a time
_lock = threading.Lock()


# ---------- chunking ----------

def clean_text(text: str) -> str:
    """Tidy whitespace, mainly for PDFs that put a line break at every visual line."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)   # single line break inside a paragraph -> space
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def split_into_chunks(text: str) -> list[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
    )
    return splitter.split_text(clean_text(text))


# ---------- embeddings and storage ----------

@lru_cache(maxsize=1)
def get_embeddings():
    """Load the embedding model once; loading takes a few seconds."""
    return HuggingFaceEmbeddings(model_name=settings.EMBEDDING_MODEL)


def _store_dir(user_id) -> Path:
    return Path(settings.VECTOR_STORE_DIR) / f"user_{user_id}"


def _load_store(user_id):
    folder = _store_dir(user_id)
    if not (folder / "index.faiss").exists():
        return None
    # allow_dangerous_deserialization is required because FAISS metadata is stored with pickle.
    # It is safe here because we only ever load files that this application wrote itself.
    return FAISS.load_local(
        str(folder), get_embeddings(), allow_dangerous_deserialization=True
    )


def _save_store(store, user_id):
    folder = _store_dir(user_id)
    folder.mkdir(parents=True, exist_ok=True)
    store.save_local(str(folder))


def _ids_for_document(store, document_id):
    prefix = f"{document_id}:"          # chunk ids look like "12:0", "12:1", ...
    return [i for i in store.index_to_docstore_id.values() if i.startswith(prefix)]


# ---------- public functions ----------

def index_document(user_id, document_id, title, text) -> int:
    """Chunk, embed and store a document. Re-indexing replaces the old chunks. Returns chunk count."""
    chunks = split_into_chunks(text)
    if not chunks:
        return 0

    docs = [
        LCDocument(
            page_content=chunk,
            metadata={"document_id": document_id, "chunk_index": i, "title": title},
        )
        for i, chunk in enumerate(chunks)
    ]
    ids = [f"{document_id}:{i}" for i in range(len(chunks))]

    with _lock:
        store = _load_store(user_id)
        if store is None:
            store = FAISS.from_documents(docs, get_embeddings(), ids=ids)
        else:
            old_ids = _ids_for_document(store, document_id)
            if old_ids:
                store.delete(old_ids)
            store.add_documents(docs, ids=ids)
        _save_store(store, user_id)
    return len(chunks)


def search(user_id, query, k=None) -> list[dict]:
    """Return the k most similar chunks from this user's documents."""
    k = k or settings.RAG_TOP_K
    with _lock:
        store = _load_store(user_id)
    if store is None:
        return []
    hits = store.similarity_search(query, k=k)
    return [
        {"document_id": h.metadata["document_id"], "chunk_text": h.page_content}
        for h in hits
    ]


def count_chunks(user_id, document_id) -> int:
    store = _load_store(user_id)
    return len(_ids_for_document(store, document_id)) if store else 0


def remove_document(user_id, document_id) -> int:
    """Delete all of a document's chunks from the index. Returns how many were removed."""
    with _lock:
        store = _load_store(user_id)
        if store is None:
            return 0
        ids = _ids_for_document(store, document_id)
        if ids:
            store.delete(ids)
            _save_store(store, user_id)
    return len(ids)