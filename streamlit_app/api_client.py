import os

import requests

BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000")
TIMEOUT = 15            # seconds, for ordinary calls
CHAT_TIMEOUT = 90       # chat can take longer: it may search documents and call the LLM
INGEST_TIMEOUT = 120    # the first ingest after a server start loads the embedding model



class ApiError(Exception):
    """A readable error from the backend, with its HTTP status when there was one."""

    def __init__(self, message, status=None):
        super().__init__(message)
        self.status = status


def _message(resp) -> str:
    try:
        data = resp.json()
    except ValueError:
        return f"Unexpected response from the server ({resp.status_code})."
    if isinstance(data, dict):
        if "detail" in data:
            return str(data["detail"])
        parts = []
        for field, errors in data.items():       # validation errors look like {"field": ["msg"]}
            if isinstance(errors, list):
                errors = " ".join(str(e) for e in errors)
            parts.append(f"{field}: {errors}")
        return "; ".join(parts)
    return str(data)


def _request(method, path, token=None, timeout=TIMEOUT, **kwargs):
    headers = {"Authorization": f"Token {token}"} if token else {}
    try:
        resp = requests.request(
            method, f"{BACKEND_URL}{path}", headers=headers, timeout=timeout, **kwargs
        )
    except requests.exceptions.RequestException:
        raise ApiError("Cannot reach the backend. Is the Django server running?")
    if resp.status_code >= 400:
        raise ApiError(_message(resp), resp.status_code)
    return resp


def login(username, password):
    return _request("POST", "/auth/login", json={"username": username, "password": password}).json()


def register(username, password):
    return _request("POST", "/auth/register", json={"username": username, "password": password}).json()


def logout(token):
    _request("POST", "/auth/logout", token=token)


def upload_document(token, filename, content, mimetype):
    return _request(
        "POST", "/documents", token=token, timeout=60,
        files={"file": (filename, content, mimetype or "application/octet-stream")},
    ).json()


def ingest_document(token, document_id):
    return _request(
        "POST", "/api/ingest-doc", token=token, timeout=INGEST_TIMEOUT,
        json={"document_id": document_id},
    ).json()


def list_documents(token):
    return _request("GET", "/documents", token=token, params={"page_size": 100}).json()["results"]

def chat(token, message, session_id):
    return _request(
        "POST", "/api/chat", token=token, timeout=CHAT_TIMEOUT,
        json={"message": message, "session_id": session_id},
    ).json()