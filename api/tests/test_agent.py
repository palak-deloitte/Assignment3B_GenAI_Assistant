import uuid
from types import SimpleNamespace

import pytest
import requests
from langchain_core.embeddings import Embeddings
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableLambda
from rest_framework.test import APIClient

from api import agent, llm, rag, weather
from api.agent import NO_DOCS_REPLY
from api.models import Document, QueryLog, UserSession

pytestmark = pytest.mark.django_db

PASSWORD = "Str0ng-Pass-123"
VOCAB = ["refund", "office", "return", "weather"]
RETURNS_TEXT = "Customers may return any product within 30 days of purchase for a full refund."
OFFICE_TEXT = "The office is open Monday to Thursday and closes at 3pm on Fridays."
REWRITTEN = "How long do refunds take for returned products?"
CITIES = ("london", "paris")


# ---------- fakes (all autouse, so no test can reach a real service) ----------

class KeywordEmbeddings(Embeddings):
    """Tiny offline stand-in for the real embedding model: one dimension per keyword."""

    def _vec(self, text):
        t = text.lower()
        return [1.0 if w in t else 0.0 for w in VOCAB] + [0.1]

    def embed_documents(self, texts):
        return [self._vec(t) for t in texts]

    def embed_query(self, text):
        return self._vec(text)


@pytest.fixture(autouse=True)
def fast_hashing(settings):
    settings.PASSWORD_HASHERS = ["django.contrib.auth.hashers.MD5PasswordHasher"]


@pytest.fixture(autouse=True)
def isolated_vector_store(settings, tmp_path, monkeypatch):
    settings.VECTOR_STORE_DIR = tmp_path / "vector_store"      # never touches your real index
    monkeypatch.setattr(rag, "get_embeddings", lambda: KeywordEmbeddings())


@pytest.fixture(autouse=True)
def fresh_agent():
    """The compiled graph (and its memory) is cached per process; start every test clean."""
    agent.get_agent.cache_clear()
    yield
    agent.get_agent.cache_clear()


def _find_city(history):
    """Fake city extraction: the most recent city named by the user in the conversation."""
    for m in reversed(history):
        if m.type != "human":
            continue
        for city in CITIES:
            if city in m.content.lower():
                return city.title()
    return "UNKNOWN"


@pytest.fixture(autouse=True)
def scripted_llm(monkeypatch):
    """A fake Claude. It recognises which prompt it was given from the system message,
    replies with a scripted answer, and records what was asked, in order."""
    rec = SimpleNamespace(kinds=[], last={})

    def respond(prompt_value):
        messages = prompt_value.to_messages()
        system, history, latest = messages[0].content, messages[1:], messages[-1].content

        if "You are a router" in system:
            kind, text = "router", latest.lower()
            if any(w in text for w in ("weather", "humidity", "temperature")):
                out = "weather"
            elif any(w in text for w in ("refund", "return", "office", "document")):
                out = "documents"
            else:
                out = "general"
        elif "Rewrite the user's latest message" in system:
            kind, out = "rewrite", REWRITTEN
        elif "Find the city" in system:
            kind, out = "city", _find_city(history)
        elif "You answer weather questions" in system:
            kind, out = "weather_answer", "WEATHER ANSWER | " + latest
        elif "using only the context" in system:
            kind, out = "rag_answer", "DOC ANSWER"
        else:
            kind, out = "general", "Hello!"

        rec.kinds.append(kind)
        rec.last[kind] = latest
        return AIMessage(content=out)

    monkeypatch.setattr(llm, "get_llm", lambda: RunnableLambda(respond))
    return rec


class FakeResponse:
    def __init__(self, status_code, data=None):
        self.status_code = status_code
        self._data = data

    def json(self):
        return self._data


@pytest.fixture(autouse=True)
def weather_http(monkeypatch, settings):
    """Fake OpenWeather. `mode` can be set to 'ok', 'not_found' or 'down'."""
    settings.OPENWEATHER_API_KEY = "test-key"
    state = SimpleNamespace(calls=[], mode="ok")

    def fake_get(url, params=None, timeout=None, **kwargs):
        state.calls.append(params["q"])
        if state.mode == "down":
            raise requests.exceptions.ConnectionError("no network")
        if state.mode == "not_found":
            return FakeResponse(404)
        return FakeResponse(200, {
            "name": params["q"],
            "main": {"temp": 14.2, "humidity": 81},
            "weather": [{"description": "overcast clouds"}],
        })

    # Replace `requests` only inside api.weather, so nothing else is affected
    monkeypatch.setattr(
        weather, "requests",
        SimpleNamespace(get=fake_get, exceptions=requests.exceptions),
    )
    return state


# ---------- users, clients, helpers ----------

@pytest.fixture
def alice(django_user_model):
    return django_user_model.objects.create_user("alice", password=PASSWORD)


@pytest.fixture
def bob(django_user_model):
    return django_user_model.objects.create_user("bob", password=PASSWORD)


def client_for(user):
    session = UserSession.objects.create(user=user)
    client = APIClient()
    client.credentials(HTTP_AUTHORIZATION=f"Token {session.token}")
    return client


@pytest.fixture
def anon():
    return APIClient()


@pytest.fixture
def alice_client(alice):
    return client_for(alice)


@pytest.fixture
def bob_client(bob):
    return client_for(bob)


def new_session():
    return uuid.uuid4().hex


def chat(client, message, session_id):
    return client.post("/api/chat", {"message": message, "session_id": session_id}, format="json")


def ingest(user, title, text):
    doc = Document.objects.create(owner=user, title=title, content=text)
    rag.index_document(user.id, doc.id, doc.title, doc.content)
    return doc


# ================= END-TO-END: message in -> tool call -> final answer =================

def test_weather_end_to_end(alice_client, scripted_llm, weather_http):
    r = chat(alice_client, "What is the weather in London right now?", new_session())
    assert r.status_code == 200
    body = r.json()

    assert body["route"] == "weather"            # the router chose the weather branch
    assert body["tool"] == "weather"             # ...and the tool was used
    assert body["sources"] == []
    assert weather_http.calls == ["London"]      # the (fake) API was called with the city
    assert "City: London" in body["answer"]      # the tool's data reached the final answer
    assert "Humidity: 81%" in body["answer"]
    assert scripted_llm.kinds == ["router", "city", "weather_answer"]


def test_documents_end_to_end(alice_client, alice, scripted_llm):
    returns = ingest(alice, "Returns policy", RETURNS_TEXT)

    r = chat(alice_client, "How many days do I have to return a product?", new_session())
    assert r.status_code == 200
    body = r.json()

    assert body["route"] == "documents"
    assert body["tool"] is None
    assert body["answer"] == "DOC ANSWER"
    assert body["sources"][0]["document_id"] == returns.id
    assert set(body["sources"][0]) == {"document_id", "chunk_text"}
    assert "30 days" in scripted_llm.last["rag_answer"]      # the retrieved chunk was in the prompt
    assert scripted_llm.kinds == ["router", "rag_answer"]    # first turn: nothing to rewrite


# ================= memory =================

def test_follow_up_uses_memory_to_find_the_city(alice_client, weather_http):
    sid = new_session()
    chat(alice_client, "What is the weather in London right now?", sid)
    r = chat(alice_client, "And what about its humidity?", sid)    # never names a city
    assert r.json()["tool"] == "weather"
    assert weather_http.calls == ["London", "London"]


def test_new_session_has_no_memory(alice_client, weather_http):
    chat(alice_client, "What is the weather in London right now?", new_session())
    r = chat(alice_client, "And what about its humidity?", new_session())
    assert r.json()["tool"] is None
    assert "Which city" in r.json()["answer"]
    assert weather_http.calls == ["London"]          # only the first question used the tool


def test_other_user_cannot_use_my_conversation(alice_client, bob_client, weather_http):
    sid = new_session()
    chat(alice_client, "What is the weather in London right now?", sid)
    r = chat(bob_client, "And what about its humidity?", sid)      # same session id, other user
    assert "Which city" in r.json()["answer"]
    assert weather_http.calls == ["London"]


def test_follow_up_question_is_rewritten_before_searching(alice_client, alice, scripted_llm, monkeypatch):
    ingest(alice, "Returns policy", RETURNS_TEXT)
    searched = []
    real_search = rag.search

    def spy(user_id, query, *args, **kwargs):
        searched.append(query)
        return real_search(user_id, query, *args, **kwargs)

    monkeypatch.setattr(rag, "search", spy)

    sid = new_session()
    chat(alice_client, "How many days do I have to return a product?", sid)
    chat(alice_client, "What about the refund time?", sid)

    assert searched[0] == "How many days do I have to return a product?"   # first turn: as typed
    assert searched[1] == REWRITTEN                                        # follow-up: rewritten first
    assert scripted_llm.kinds == ["router", "rag_answer", "router", "rewrite", "rag_answer"]


def test_sources_do_not_leak_into_the_next_turn(alice_client, alice):
    ingest(alice, "Returns policy", RETURNS_TEXT)
    sid = new_session()
    assert chat(alice_client, "How many days do I have to return a product?", sid).json()["sources"]
    second = chat(alice_client, "Thanks!", sid).json()
    assert second["route"] == "general"
    assert second["sources"] == []


def test_tool_flag_does_not_leak_into_the_next_turn(alice_client, weather_http):
    sid = new_session()
    assert chat(alice_client, "What is the weather in London right now?", sid).json()["tool"] == "weather"
    assert chat(alice_client, "Thanks!", sid).json()["tool"] is None


# ================= routing (the conditional edges) =================

@pytest.mark.parametrize("message, route", [
    ("Hi there!", "general"),
    ("What is the weather in London right now?", "weather"),
    ("How many days do I have to return a product?", "documents"),
])
def test_router_sends_each_kind_of_message_to_a_different_branch(alice_client, message, route):
    assert chat(alice_client, message, new_session()).json()["route"] == route


def test_general_chat_uses_no_tools(alice_client, scripted_llm, weather_http):
    r = chat(alice_client, "Hi there!", new_session())
    assert r.json() == {"answer": "Hello!", "sources": [], "route": "general", "tool": None}
    assert weather_http.calls == []
    assert scripted_llm.kinds == ["router", "general"]


def test_documents_route_can_use_several_documents(alice_client, alice):
    a = ingest(alice, "Returns policy", RETURNS_TEXT)
    b = ingest(alice, "Office hours", OFFICE_TEXT)
    r = chat(alice_client, "When does the office close and how long do refunds take?", new_session())
    assert {s["document_id"] for s in r.json()["sources"]} == {a.id, b.id}


def test_agent_never_searches_other_users_documents(bob_client, alice):
    ingest(alice, "Returns policy", RETURNS_TEXT)
    r = chat(bob_client, "How many days do I have to return a product?", new_session())
    assert r.json()["answer"] == NO_DOCS_REPLY
    assert r.json()["sources"] == []


# ================= the weather tool: failure handling =================

def test_unknown_city_is_reported_politely(alice_client, weather_http):
    weather_http.mode = "not_found"
    r = chat(alice_client, "What is the weather in London right now?", new_session())
    assert r.status_code == 200
    assert "couldn't find a city called London" in r.json()["answer"]
    assert r.json()["tool"] is None


def test_weather_service_down_is_handled(alice_client, weather_http):
    weather_http.mode = "down"
    r = chat(alice_client, "What is the weather in London right now?", new_session())
    assert r.status_code == 200
    assert "isn't available right now" in r.json()["answer"]


def test_weather_not_configured_is_handled(alice_client, weather_http, settings):
    settings.OPENWEATHER_API_KEY = None
    r = chat(alice_client, "What is the weather in London right now?", new_session())
    assert r.status_code == 200
    assert "isn't available right now" in r.json()["answer"]
    assert weather_http.calls == []


def test_weather_question_without_a_city_asks_for_one(alice_client, weather_http):
    r = chat(alice_client, "What's the weather like?", new_session())
    assert "Which city" in r.json()["answer"]
    assert weather_http.calls == []


# ================= small pure functions =================

@pytest.mark.parametrize("text, expected", [
    ("weather", "weather"),
    ("  Documents. ", "documents"),
    ("general", "general"),
    ("I think maybe weather?", "weather"),
    ("no idea", "general"),
    ("", "general"),
])
def test_parse_route(text, expected):
    assert agent.parse_route(text) == expected


@pytest.mark.parametrize("text, expected", [
    ("London", "London"),
    ('"Paris".', "Paris"),
    ("UNKNOWN", None),
    ("unknown", None),
    ("", None),
    ("x" * 61, None),
    ("Paris\nFrance", None),
])
def test_parse_city(text, expected):
    assert agent.parse_city(text) == expected


def test_history_starts_with_a_human_message():
    msgs = [AIMessage(content="a"), HumanMessage(content="h"), AIMessage(content="b")]
    assert [m.content for m in agent._history({"messages": msgs})] == ["h", "b"]


def test_history_is_trimmed():
    msgs = [HumanMessage(content=str(i)) if i % 2 == 0 else AIMessage(content=str(i)) for i in range(12)]
    trimmed = agent._history({"messages": msgs})
    assert len(trimmed) <= agent.MAX_HISTORY
    assert trimmed[0].type == "human"


# ================= the /api/chat endpoint =================

def test_chat_turn_is_logged_in_history(alice_client, alice):
    chat(alice_client, "Hi there!", new_session())
    log = QueryLog.objects.get(user=alice)
    assert log.kind == QueryLog.Kind.CHAT
    assert log.query_text == "Hi there!"
    assert log.result_summary == "Hello!"
    items = alice_client.get("/api/query-history").json()["results"]
    assert [i["kind"] for i in items] == ["chat"]


@pytest.mark.parametrize("payload", [
    {"message": "", "session_id": "session-12345"},
    {"message": "   ", "session_id": "session-12345"},
    {"session_id": "session-12345"},
    {"message": "hi"},
    {"message": "hi", "session_id": "x"},
    {"message": "hi", "session_id": "bad id with spaces!"},
])
def test_chat_rejects_invalid_payload(alice_client, scripted_llm, payload):
    assert alice_client.post("/api/chat", payload, format="json").status_code == 400
    assert scripted_llm.kinds == []              # the model was never called
    assert QueryLog.objects.count() == 0


def test_chat_rejects_a_very_long_message(alice_client):
    assert chat(alice_client, "x" * 2001, new_session()).status_code == 400


def test_chat_requires_token(anon):
    assert chat(anon, "hi", new_session()).status_code == 401


def test_llm_failure_is_502_and_not_logged(alice_client, monkeypatch):
    def failing(_):
        raise RuntimeError("rate limited")

    monkeypatch.setattr(llm, "get_llm", lambda: RunnableLambda(failing))
    r = chat(alice_client, "Hi there!", new_session())
    assert r.status_code == 502
    assert QueryLog.objects.count() == 0


def test_retrieval_failure_is_503(alice_client, monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError("index corrupted")

    monkeypatch.setattr(rag, "search", boom)
    r = chat(alice_client, "How many days do I have to return a product?", new_session())
    assert r.status_code == 503


def test_unexpected_agent_failure_is_503(alice_client, monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError("something broke")

    monkeypatch.setattr(agent, "run_agent", boom)
    assert chat(alice_client, "hi", new_session()).status_code == 503