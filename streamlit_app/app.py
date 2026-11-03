import uuid

import streamlit as st

import api_client
from api_client import ApiError

st.set_page_config(page_title="Document Assistant", page_icon="📄")


# ---------- state ----------

def init_state():
    # Streamlit re-runs this whole script on every click, so anything that must
    # survive between clicks lives in st.session_state.
    st.session_state.setdefault("token", None)
    st.session_state.setdefault("username", None)
    st.session_state.setdefault("documents", None)        # None = not loaded yet
    st.session_state.setdefault("uploader_key", 0)        # changing it resets the file picker
    st.session_state.setdefault("upload_results", [])
    st.session_state.setdefault("messages", [])           # the conversation, for display
    st.session_state.setdefault("session_id", str(uuid.uuid4()))   # the agent's memory key


def sign_in(data, username):
    st.session_state.token = data["token"]
    st.session_state.username = username
    st.rerun()


def expire_and_restart():
    st.session_state.clear()
    st.session_state["flash"] = "Your session expired. Please log in again."
    st.rerun()


def refresh_documents():
    try:
        st.session_state.documents = api_client.list_documents(st.session_state.token)
    except ApiError as e:
        if e.status == 401:
            expire_and_restart()
        st.session_state.documents = []


def new_conversation():
    # A new session_id means a new memory thread on the backend
    st.session_state.messages = []
    st.session_state.session_id = str(uuid.uuid4())


# ---------- login screen ----------

def show_auth():
    st.title("📄 Document Assistant")
    if "flash" in st.session_state:
        st.warning(st.session_state.pop("flash"))
    st.caption("Log in to upload documents and chat with them.")
    login_tab, register_tab = st.tabs(["Log in", "Register"])

    with login_tab:
        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Log in")
        if submitted:
            try:
                sign_in(api_client.login(username, password), username)
            except ApiError as e:
                st.error(str(e))

    with register_tab:
        with st.form("register_form"):
            new_username = st.text_input("Choose a username")
            new_password = st.text_input("Choose a password", type="password")
            submitted = st.form_submit_button("Create account")
        if submitted:
            try:
                sign_in(api_client.register(new_username, new_password), new_username)
            except ApiError as e:
                st.error(str(e))


# ---------- uploads ----------

def process_uploads(files):
    """Upload and ingest each file. Returns (kind, text) messages to show afterwards."""
    token = st.session_state.token
    results = []
    for f in files:
        with st.spinner(f"Processing {f.name} (the first one can take a while)..."):
            try:
                doc = api_client.upload_document(token, f.name, f.getvalue(), f.type)
            except ApiError as e:
                if e.status == 401:
                    expire_and_restart()
                results.append(("error", f"❌ {f.name}: {e}"))
                continue
            try:
                info = api_client.ingest_document(token, doc["id"])
            except ApiError as e:
                if e.status == 401:
                    expire_and_restart()
                results.append(("warning", f"⚠️ {f.name} was uploaded but could not be indexed: {e}"))
                continue
        results.append(("success", f"✅ {f.name}: ingested into {info['chunk_count']} chunk(s)"))
    return results


def show_sidebar():
    with st.sidebar:
        st.write(f"Signed in as **{st.session_state.username}**")
        if st.button("Log out"):
            try:
                api_client.logout(st.session_state.token)
            except ApiError:
                pass                      # the token may already be expired; log out locally anyway
            st.session_state.clear()
            st.rerun()

        if st.button("🗑️ New conversation"):
            new_conversation()
            st.rerun()

        st.divider()
        st.subheader("Upload documents")
        files = st.file_uploader(
            "PDF or TXT, up to 5 MB each",
            type=["pdf", "txt"],
            accept_multiple_files=True,
            key=f"uploader_{st.session_state.uploader_key}",
        )
        if st.button("Upload & ingest", disabled=not files):
            st.session_state.upload_results = process_uploads(files)
            st.session_state.uploader_key += 1        # clears the file picker
            refresh_documents()
            st.rerun()

        for kind, text in st.session_state.upload_results:
            getattr(st, kind)(text)

        st.divider()
        docs = st.session_state.documents or []
        st.subheader(f"Your documents ({len(docs)})")
        if not docs:
            st.caption("No documents yet.")
        for d in docs:
            st.caption(f"📄 {d['title']}")


# ---------- chat ----------

def show_meta(msg):
    """What the agent did, plus the source chunks it used."""
    if msg.get("tool") == "weather":
        st.caption("🌤️ Used the weather tool")
    elif msg.get("route") == "documents":
        st.caption("📚 Searched your documents")

    sources = msg.get("sources") or []
    if sources:
        titles = {d["id"]: d["title"] for d in st.session_state.documents or []}
        with st.expander(f"Sources ({len(sources)})"):
            for i, s in enumerate(sources, start=1):
                title = titles.get(s["document_id"], f"Document {s['document_id']}")
                st.markdown(f"**[{i}] {title}**")
                st.caption(s["chunk_text"])


def render_message(msg):
    with st.chat_message(msg["role"]):
        if msg.get("error"):
            st.error(msg["content"])
        else:
            st.markdown(msg["content"])
            if msg["role"] == "assistant":
                show_meta(msg)


def handle_prompt(prompt):
    user_msg = {"role": "user", "content": prompt}
    st.session_state.messages.append(user_msg)
    render_message(user_msg)

    with st.spinner("Thinking..."):
        try:
            result = api_client.chat(
                st.session_state.token, prompt, st.session_state.session_id
            )
            reply = {
                "role": "assistant",
                "content": result["answer"],
                "sources": result["sources"],
                "route": result["route"],
                "tool": result["tool"],
            }
        except ApiError as e:
            if e.status == 401:
                expire_and_restart()
            reply = {"role": "assistant", "content": str(e), "error": True}

    st.session_state.messages.append(reply)
    render_message(reply)


# ---------- main screen ----------

def show_app():
    if st.session_state.documents is None:
        refresh_documents()
    show_sidebar()

    st.title("📄 Document Assistant")
    if not st.session_state.messages:
        st.caption(
            "Ask about your uploaded documents, or about the weather in any city. "
            "Try: *What does my document say about refunds?* or *What's the weather in Paris?*"
        )

    for msg in st.session_state.messages:          # redraw the whole conversation
        render_message(msg)

    prompt = st.chat_input("Ask about your documents or the weather", max_chars=2000)
    if prompt:
        handle_prompt(prompt)


init_state()
if st.session_state.token:
    show_app()
else:
    show_auth()