"""
NN's PDF Q&A Chatbot (LangChain 0.3+ compatible)
- Gemini embeddings via langchain-google-genai (embedding model hard-coded)
- Chroma local vector DB via langchain_community.vectorstores
- Groq LLM (SDK if present, otherwise HTTP fallback)
Keys are entered at runtime in the Streamlit sidebar (no .env required).
"""

import os
import tempfile
import json
from typing import List

import streamlit as st
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Chroma wrapper from langchain community package
try:
    from langchain_community.vectorstores import Chroma
except Exception:
    Chroma = None

# Gemini embeddings via LangChain Google GenAI wrapper
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

import requests

# UI config
st.set_page_config(page_title="NN's PDF Q&A Chatbot", layout="wide")

# --- Styling: radiant background with golden-orange sidebar
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(135deg, #3a0ca3 0%, #7209b7 50%, #4b0082 100%);
        color: white;
    }
    /* Sidebar background: vibrant golden orange */
    .stSidebar .sidebar-content {
        background: linear-gradient(180deg, #ffb703 0%, #fb8500 100%);
        color: white;
        padding: 16px;
        border-radius: 8px;
    }
    .stTextInput>div>div>input, .stTextArea>div>div>textarea {
        background-color: rgba(255,255,255,0.06);
        color: white;
    }
    .stButton>button { background-color: rgba(255,255,255,0.08); color: white; }
    h1, h2, h3, h4 { color: white; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("NN's PDF Q&A Chatbot")
st.write("Upload a PDF, embed with Gemini embeddings, and query using Groq. Enter keys in the sidebar.")

# --------------------------
# Sidebar: Groq & Gemini keys + model dropdown
# --------------------------
with st.sidebar.form("settings"):
    st.header("Runtime keys & model")
    st.write("Enter keys here. They are stored only in session state while app runs.")
    groq_api_key = st.text_input("Groq API Key", type="password")
    google_api_key = st.text_input("Gemini (Google) API Key", type="password")

    # Dropdown: three example Groq model names (replace if your account uses different names)
    groq_model = st.selectbox(
        "Groq model",
        options=[
            "chat.groq",          # placeholder / common example
            "groq-mini",          # example smaller model
            "groq-instruct"       # example instruct-style
        ],
        index=0,
    )

    st.caption("If your Groq account exposes different model names, pick the matching one.")
    submitted = st.form_submit_button("Save")

if submitted:
    st.session_state["groq_api_key"] = groq_api_key
    st.session_state["google_api_key"] = google_api_key
    st.session_state["groq_model"] = groq_model
    st.success("Saved to session state (runtime only).")

# Load session values if set
groq_api_key = st.session_state.get("groq_api_key", "")
google_api_key = st.session_state.get("google_api_key", "")
groq_model = st.session_state.get("groq_model", groq_model if 'groq_model' in locals() else "chat.groq")

# Fixed top_k
TOP_K = 4

# Warnings for missing optional dependencies
if not GOOGLE_GENAI_AVAILABLE:
    st.sidebar.error("Missing 'langchain-google-genai'. Install it to use Gemini embeddings.")
if Chroma is None:
    st.sidebar.error("Missing Chroma vectorstore wrapper. Install 'langchain_community' or appropriate package.")
if not GROQ_SDK_AVAILABLE:
    st.sidebar.info("Groq SDK not installed; the app will attempt HTTP fallback to Groq API if needed.")

# --------------------------
# File uploader and pipeline
# --------------------------
uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

vectorstore = None
texts: List[str] = []

# Hard-coded embedding model (Gemini embedding choice)
EMBEDDING_MODEL = "gemini-embedding-001"

def create_vectorstore_from_texts(texts: List[str], google_api_key: str):
    """
    Create a Chroma vectorstore from texts using Gemini embeddings.
    Returns Chroma vectorstore instance or raises error.
    """
    if not GOOGLE_GENAI_AVAILABLE or GoogleGenerativeAIEmbeddings is None:
        raise RuntimeError("Gemini embeddings package not installed or importable.")
    if Chroma is None:
        raise RuntimeError("Chroma vectorstore wrapper is not available.")
    # Initialize embeddings wrapper (wrapper param name differs across versions; this code expects google_api_key & model)
    embeddings = GoogleGenerativeAIEmbeddings(google_api_key=google_api_key, model=EMBEDDING_MODEL)
    # Chroma.from_texts computes embeddings and returns a vectorstore
    vs = Chroma.from_texts(texts=texts, embedding=embeddings)
    return vs

if uploaded_file:
    status = st.empty()
    status.info("Uploading PDF...")

    # Save temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    status.info("Extracting text from PDF...")
    try:
        reader = PdfReader(tmp_path)
        raw_text = ""
        for page in reader.pages:
            t = page.extract_text()
            if t:
                raw_text += t + "\n"
    except Exception as e:
        st.error(f"PDF read error: {e}")
        raise

    if not raw_text.strip():
        st.error("No extractable text found in PDF.")
    else:
        status.info("Splitting text into chunks...")
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=200,
            separators=["\n\n", "\n", " ", ""],
        )
        texts = splitter.split_text(raw_text)
        status.info(f"Created {len(texts)} chunks — embedding with Gemini...")

        if not google_api_key:
            st.error("Gemini API key required. Enter it in the sidebar and re-run embedding step.")
            st.stop()

        try:
            vectorstore = create_vectorstore_from_texts(texts, google_api_key)
        except Exception as e:
            st.error(f"Failed to create vectorstore: {e}")
            st.stop()

        status.success("Document embedded successfully — ready to query.")
        st.markdown("---")
        st.subheader("Ask questions about the uploaded PDF")

        question = st.text_input("Enter your question:")

        if st.button("Get answer") and question.strip():
            if not groq_api_key:
                st.error("Groq API key required. Enter it in the sidebar.")
                st.stop()

            # Retrieve documents (try different retrieval methods based on wrapper implementation)
            try:
                if hasattr(vectorstore, "similarity_search_with_score"):
                    hits = vectorstore.similarity_search_with_score(question, k=TOP_K)
                    docs = [d for d, _ in hits]
                    scores = [s for _, s in hits]
                elif hasattr(vectorstore, "similarity_search"):
                    hits = vectorstore.similarity_search(question, k=TOP_K)
                    docs = hits
                    scores = None
                else:
                    raise RuntimeError("Vectorstore retrieval method not available.")
            except Exception as e:
                st.error(f"Retrieval error: {e}")
                st.stop()

            # Build context (truncate each snippet to avoid huge prompts)
            def safe_snippet(s: str, max_chars: int = 2000):
                return s.replace("\n", " ")[:max_chars]

            context_parts = [safe_snippet(getattr(d, "page_content", str(d))) for d in docs]
            context_text = "\n\n---\n\n".join(context_parts)

            prompt = (
                "You are a helpful assistant. Use the document context below to answer the question. "
                "If the answer cannot be found in the document context, reply: \"I don't know based on the provided document.\""
                f"\n\nContext:\n{context_text}\n\nQuestion: {question}\n\nAnswer:"
            )

            st.info("Querying Groq LLM for answer...")

            answer_text = None
            groq_error = None

            # Try Groq SDK first (if available)
            if GROQ_SDK_AVAILABLE and Groq is not None:
                try:
                    client = Groq(api_key=groq_api_key)
                    try:
                        response = client.chat.create(messages=[{"role": "user", "content": prompt}], model=groq_model)
                        if isinstance(response, dict):
                            # Common shapes: response["content"] or OpenAI-like response["choices"][0]["message"]["content"]
                            if response.get("content"):
                                answer_text = response["content"]
                            elif "choices" in response and len(response["choices"]) > 0:
                                ch = response["choices"][0]
                                if isinstance(ch, dict) and "message" in ch and "content" in ch["message"]:
                                    answer_text = ch["message"]["content"]
                                elif "text" in ch:
                                    answer_text = ch["text"]
                                else:
                                    answer_text = str(ch)
                            else:
                                answer_text = str(response)
                        else:
                            answer_text = str(response)
                    except Exception:
                        # fallback to other SDK shapes
                        try:
                            response = client.completions.create(prompt=prompt, model=groq_model)
                            if isinstance(response, dict) and "choices" in response:
                                answer_text = response["choices"][0].get("text", "")
                            else:
                                answer_text = str(response)
                        except Exception as e:
                            groq_error = f"Groq SDK call failed: {e}"
                except Exception as e:
                    groq_error = f"Groq SDK initialization failed: {e}"

            # If no answer_text yet, try HTTP fallback to Groq REST endpoint
            if not answer_text:
                try:
                    headers = {
                        "Authorization": f"Bearer {groq_api_key}",
                        "Content-Type": "application/json",
                    }
                    payload = {
                        "model": groq_model,
                        "messages": [{"role": "user", "content": prompt}],
                        "max_tokens": 800,
                        "temperature": 0.0,
                    }
                    BASE_URL = "https://api.groq.ai/v1"  # update if Groq uses another base URL for your account
                    endpoint = f"{BASE_URL}/chat/completions"
                    resp = requests.post(endpoint, headers=headers, json=payload, timeout=30)
                    if resp.status_code == 200:
                        j = resp.json()
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
                        groq_error = f"Groq HTTP failure ({resp.status_code}): {resp.text}"
                except Exception as e:
                    groq_error = f"Groq HTTP fallback exception: {e}"

            if not answer_text:
                st.error("Failed to obtain answer from Groq.")
                if groq_error:
                    st.write("Details:", groq_error)
                st.stop()

            st.success("ANSWER")
            st.write(answer_text)

            st.markdown("**Top documents (by relevance)**")
            if scores:
                for d, s in zip(docs, scores):
                    snippet = getattr(d, "page_content", str(d))[:700].replace("\n", " ")
                    st.write(f"[{s:.4f}] {snippet}...")
            else:
                for i, d in enumerate(docs, start=1):
                    snippet = getattr(d, "page_content", str(d))[:700].replace("\n", " ")
                    st.write(f"[{i}] {snippet}...")

# Footer
st.markdown("---")
st.caption(
    "Notes: Embeddings use Gemini (hard-coded model: gemini-embedding-001). Groq models in the dropdown are examples — "
    "replace with your account's supported model names if different. "
    "If you need a persistent vector store (Astra, Pinecone, Milvus) or an Astra-enabled ZIP with .env, request the Astra variant."
)
