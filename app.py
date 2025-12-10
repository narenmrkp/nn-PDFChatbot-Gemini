"""
NN's PDF Q&A Chatbot (LangChain 0.3+ compatible)
- Gemini embeddings via langchain-google-genai
- Chroma local vector DB via langchain_community.vectorstores
- Groq LLM (SDK if present, otherwise HTTP fallback)
Keys are entered at runtime in the Streamlit sidebar (no .env required).
"""

import os
import tempfile
import json
import time
from typing import List, Tuple

import streamlit as st
from PyPDF2 import PdfReader

# LangChain 0.3+ modular imports
from langchain_text_splitters import RecursiveCharacterTextSplitter

try:
    # Chroma integration for LangChain community package
    from langchain_community.vectorstores import Chroma
except Exception:
    Chroma = None

# Embeddings (Gemini via langchain-google-genai)
try:
    from langchain_google_genai import GoogleGenerativeAIEmbeddings
    GOOGLE_GENAI_AVAILABLE = True
except Exception:
    GoogleGenerativeAIEmbeddings = None
    GOOGLE_GENAI_AVAILABLE = False

# Groq SDK attempt
try:
    from groq import Groq
    GROQ_SDK_AVAILABLE = True
except Exception:
    Groq = None
    GROQ_SDK_AVAILABLE = False

import requests  # for HTTP fallback to Groq

st.set_page_config(page_title="NN's PDF Q&A Chatbot", layout="wide")

# --- Styling (radiant blue-violet background, white text)
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(135deg, #3a0ca3 0%, #7209b7 50%, #4b0082 100%);
        color: white;
    }
    .stSidebar .sidebar-content { background: transparent; color: white; }
    .stTextInput>div>div>input, .stTextArea>div>div>textarea {
        background-color: rgba(255,255,255,0.06);
        color: white;
    }
    .stButton>button { background-color: rgba(255,255,255,0.08); color: white; }
    h1, h2, h3, h4, h5, h6 { color: white; }
    .css-1v3fvcr { color: white; } /* extra fallback */
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("NN's PDF Q&A Chatbot")
st.write("Upload a PDF, embed with Gemini embeddings, and query using Groq LLM. Enter keys in the sidebar.")

# --------------------------
# Sidebar: Keys and settings
# --------------------------
with st.sidebar.form("settings"):
    st.header("Keys & Settings (runtime only)")
    st.write("Enter API keys below. Keys are stored only in Streamlit session state.")
    groq_api_key = st.text_input("Groq API Key", type="password")
    groq_model = st.text_input("Groq model name", value="chat.groq")
    google_api_key = st.text_input("Google (Gemini/Vertex AI) API Key", type="password")
    embedding_model = st.text_input("Gemini embedding model", value="gemini-embedding-001")
    top_k = st.number_input("Top-k documents to retrieve", min_value=1, max_value=10, value=4)
    submit_settings = st.form_submit_button("Save settings")

if submit_settings:
    st.session_state['groq_api_key'] = groq_api_key
    st.session_state['groq_model'] = groq_model
    st.session_state['google_api_key'] = google_api_key
    st.session_state['embedding_model'] = embedding_model
    st.session_state['top_k'] = int(top_k)
    st.success("Settings saved in session state.")

# Load from session state (if previously saved)
groq_api_key = st.session_state.get('groq_api_key', "")
groq_model = st.session_state.get('groq_model', "chat.groq")
google_api_key = st.session_state.get('google_api_key', "")
embedding_model = st.session_state.get('embedding_model', "gemini-embedding-001")
top_k = st.session_state.get('top_k', 4)

# Dependency warnings
if not GOOGLE_GENAI_AVAILABLE:
    st.sidebar.error("Missing dependency: install 'langchain-google-genai' to enable Gemini embeddings.")
if Chroma is None:
    st.sidebar.error("Missing dependency: install 'langchain-community' (or the package providing Chroma wrapper).")
if not GROQ_SDK_AVAILABLE:
    st.sidebar.info("Groq SDK not found; the app will try HTTP fallback to Groq API. Install 'groq' SDK for better integration.")

# --------------------------
# File uploader & processing
# --------------------------
uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

# Local runtime objects
vectorstore = None
texts: List[str] = []
doc_metadatas = []

if uploaded_file is not None:
    status = st.empty()
    status.info("Document uploading...")

    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    status.info("PDF uploaded successfully — extracting text...")

    # Extract text
    try:
        reader = PdfReader(tmp_path)
        raw_text = ""
        for p in reader.pages:
            page_text = p.extract_text()
            if page_text:
                raw_text += page_text + "\n"
    except Exception as e:
        st.error(f"Failed to read PDF: {e}")
        raise

    if not raw_text.strip():
        st.error("No extractable text found in the uploaded PDF.")
    else:
        status.info("Text extracted — splitting into chunks...")

        # Use recursive splitter (LangChain 0.3+ compatible)
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=200,
            separators=["\n\n", "\n", " ", ""],
        )
        texts = splitter.split_text(raw_text)
        status.info(f"{len(texts)} chunks created — preparing embeddings...")

        # Ensure Google key present
        if not google_api_key:
            st.error("Google API key is required for Gemini embeddings. Enter it in the sidebar and re-embed.")
            st.stop()

        if not GOOGLE_GENAI_AVAILABLE or GoogleGenerativeAIEmbeddings is None:
            st.error("Gemini embeddings integration (langchain-google-genai) is not installed or importable.")
            st.stop()

        # Initialize embeddings
        try:
            embeddings = GoogleGenerativeAIEmbeddings(google_api_key=google_api_key, model=embedding_model)
        except Exception as e:
            st.error(f"Failed to initialize Gemini embeddings: {e}")
            st.stop()

        # Build a Chroma vectorstore (local)
        if Chroma is None:
            st.error("Chroma vectorstore is not available in this environment. Install the required package.")
            st.stop()

        try:
            # Chroma.from_texts will compute embeddings and index them
            # Use a small in-memory collection (no persistence by default)
            vectorstore = Chroma.from_texts(texts=texts, embedding=embeddings)
        except Exception as e:
            st.error(f"Failed to create Chroma vectorstore: {e}")
            st.stop()

        status.success("Document embedded successfully and ready to query.")
        st.markdown("---")

        # Query UI
        st.subheader("Ask questions")
        question = st.text_input("Enter your question about the uploaded PDF:")

        if st.button("Get answer") and question.strip():
            if not groq_api_key:
                st.error("Groq API key is required. Enter it in the sidebar.")
                st.stop()

            # Retrieve top-k similar documents
            try:
                # The Chroma wrapper's retrieval method name can vary; try common ones
                if hasattr(vectorstore, "similarity_search"):
                    hits = vectorstore.similarity_search(question, k=top_k)
                    # hits might be a list of Document objects or tuples
                    if hits and isinstance(hits[0], tuple):
                        docs = [h[0] for h in hits]
                    else:
                        docs = hits
                    # If scores are not returned, set as None
                    scores = None
                elif hasattr(vectorstore, "similarity_search_with_score"):
                    hits = vectorstore.similarity_search_with_score(question, k=top_k)
                    docs = [d for d, _ in hits]
                    scores = [s for _, s in hits]
                else:
                    raise RuntimeError("Vectorstore retrieval method not found.")
            except Exception as e:
                st.error(f"Failed to retrieve documents from vectorstore: {e}")
                st.stop()

            # Compile context snippet (truncate per-snippet to a safe token length)
            def safe_snippet(text: str, max_chars: int = 2000) -> str:
                return text[:max_chars].replace("\n", " ")

            context_list = [safe_snippet(getattr(d, "page_content", str(d))) for d in docs]
            context_text = "\n\n---\n\n".join(context_list)

            # Build the prompt
            prompt = (
                "You are an assistant that answers questions using the provided PDF context. "
                "If the answer is not present in the context, reply: \"I don't know based on the provided document.\" "
                "\n\n"
                "Context:\n"
                f"{context_text}\n\n"
                f"Question: {question}\n\nAnswer:"
            )

            st.info("Querying Groq LLM — composing answer from retrieved context...")

            # --------------------
            # Groq call (SDK if available, else HTTP fallback)
            # --------------------
            answer_text = None
            groq_error = None

            # First attempt: groq SDK
            if GROQ_SDK_AVAILABLE and Groq is not None:
                try:
                    client = Groq(api_key=groq_api_key)
                    # The official SDK API shape may differ. Attempt to call a chat-like endpoint.
                    # Many official SDKs use client.chat.create(...) with messages.
                    # We'll try a few shapes defensively.
                    try:
                        response = client.chat.create(messages=[{"role": "user", "content": prompt}], model=groq_model)
                        # response parsing varies by SDK; attempt several common patterns
                        if isinstance(response, dict):
                            # try common keys
                            if response.get("content"):
                                answer_text = response["content"]
                            elif response.get("choices"):
                                # OpenAI-like shape
                                choice = response["choices"][0]
                                if isinstance(choice, dict) and "message" in choice:
                                    answer_text = choice["message"].get("content", "")
                                else:
                                    answer_text = str(choice)
                            else:
                                answer_text = str(response)
                        else:
                            # fallback to string
                            answer_text = str(response)
                    except Exception:
                        # some SDKs provide client.completions.create or client.create_chat_completion
                        try:
                            response = client.completions.create(prompt=prompt, model=groq_model)
                            # parse typical shape
                            if isinstance(response, dict) and "choices" in response:
                                answer_text = response["choices"][0]["text"]
                            else:
                                answer_text = str(response)
                        except Exception as e:
                            # leave to fallback below
                            groq_error = f"Groq SDK call failed: {e}"
                except Exception as e:
                    groq_error = f"Failed to initialize or call Groq SDK: {e}"

            # HTTP fallback (generic)
            if not answer_text:
                # Try an HTTP POST to Groq's REST endpoint (if accessible). This fallback requires the proper endpoint.
                # NOTE: Replace the endpoint if your Groq account uses a different base URL.
                try:
                    headers = {
                        "Authorization": f"Bearer {groq_api_key}",
                        "Content-Type": "application/json",
                    }
                    # Common Groq REST path (may vary by provider account)
                    # We attempt a chat/completions-like request; adjust if your API expects different fields.
                    payload = {
                        "model": groq_model,
                        "messages": [{"role": "user", "content": prompt}],
                        "max_tokens": 800,
                        "temperature": 0.0,
                    }
                    # Default base URL (may not match your account). If this fails, you will need to replace BASE_URL.
                    BASE_URL = "https://api.groq.ai/v1"  # please update if Groq provides a different endpoint
                    endpoint = f"{BASE_URL}/chat/completions"
                    resp = requests.post(endpoint, headers=headers, data=json.dumps(payload), timeout=30)
                    if resp.status_code == 200:
                        j = resp.json()
                        # try parse OpenAI-like response
                        if "choices" in j and isinstance(j["choices"], list) and len(j["choices"]) > 0:
                            c = j["choices"][0]
                            if isinstance(c, dict) and "message" in c and "content" in c["message"]:
                                answer_text = c["message"]["content"]
                            elif "text" in c:
                                answer_text = c["text"]
                            else:
                                answer_text = str(c)
                        elif "content" in j:
                            answer_text = j["content"]
                        else:
                            answer_text = json.dumps(j)
                    else:
                        groq_error = f"Groq HTTP request failed (status {resp.status_code}): {resp.text}"
                except Exception as e:
                    groq_error = f"Groq HTTP fallback failed: {e}"

            if not answer_text:
                st.error("Failed to obtain answer from Groq.")
                if groq_error:
                    st.write("Details:", groq_error)
                st.stop()

            # Display answer & top docs
            st.success("ANSWER")
            st.write(answer_text)

            st.markdown("**Top documents (by relevance)**")
            # show doc snippets and optional scores if available
            if scores:
                for d, s in zip(docs, scores):
                    snippet = getattr(d, "page_content", str(d))[:700].replace("\n", " ")
                    st.write(f"[{s:.4f}] {snippet}...")
            else:
                for i, d in enumerate(docs, start=1):
                    snippet = getattr(d, "page_content", str(d))[:700].replace("\n", " ")
                    st.write(f"[{i}] {snippet}...")

# Footer notes
st.markdown("---")
st.caption(
    "Notes: This app uses Gemini embeddings via langchain-google-genai and Chroma locally for vector storage. "
    "Groq integration uses the official SDK if available; otherwise an HTTP fallback is attempted. "
    "If Groq API endpoints or model names differ for your account, update the 'Groq model name' and BASE_URL as needed."
)
