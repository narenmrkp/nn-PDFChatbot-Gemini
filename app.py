"""
NN's PDF Q&A Chatbot — Updated
- Sidebar: Groq API Key (masked), Gemini API Key (masked), Groq model dropdown
- Embeddings: Gemini (gemini-embedding-001) — key validated with a quick test embed
- LLM: Groq (SDK if available), HTTP fallback if required
- Vectorstore: Chroma local
- All UI text (including errors) forced to white for readable contrast on violet background
"""

import os
import tempfile
import json
from typing import List, Optional

import streamlit as st
from PyPDF2 import PdfReader

# LangChain 0.3+ splitter
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Chroma wrapper (langchain community)
try:
    from langchain_community.vectorstores import Chroma
except Exception:
    Chroma = None

# Gemini embeddings wrapper
try:
    from langchain_google_genai import GoogleGenerativeAIEmbeddings
    GOOGLE_GENAI_AVAILABLE = True
except Exception:
    GoogleGenerativeAIEmbeddings = None
    GOOGLE_GENAI_AVAILABLE = False

# Groq SDK
try:
    from groq import Groq
    GROQ_SDK_AVAILABLE = True
except Exception:
    Groq = None
    GROQ_SDK_AVAILABLE = False

import requests

# UI config
st.set_page_config(page_title="NN's PDF Q&A Chatbot", layout="wide")

# ---------- CSS: violet background + white text everywhere + golden-orange sidebar ----------
st.markdown(
    """
    <style>
    /* App background */
    .stApp {
        background: linear-gradient(135deg, #3a0ca3 0%, #7209b7 50%, #4b0082 100%) !important;
        color: white !important;
    }

    /* Sidebar vibrant golden orange */
    .stSidebar .sidebar-content {
        background: linear-gradient(180deg, #ffb703 0%, #fb8500 100%) !important;
        color: white !important;
        padding: 16px !important;
        border-radius: 8px !important;
    }

    /* Force text color white in many common elements (covers messages, markdown, results) */
    .stTextInput, .stTextArea, .stMarkdown, .stButton, .stExpander, .stMetric {
        color: white !important;
    }

    /* Inputs and textareas readable */
    input, textarea {
        color: white !important;
        background-color: rgba(255,255,255,0.06) !important;
    }

    /* Force Streamlit alert/exception look to white text for readability */
    .stAlert, .stException, .stError, .stWarning {
        color: white !important;
        background: rgba(0,0,0,0.18) !important;
    }

    /* Make outputs (markdown) white */
    .streamlit-expanderHeader, .stMarkdown, .stText {
        color: white !important;
    }

    /* Safety: ensure small fonts for captions remain visible */
    .stCaption {
        color: white !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("NN's PDF Q&A Chatbot")
st.write("Upload a PDF, embed with Gemini embeddings, and query using Groq. Enter keys in the sidebar.")

# ---------- Sidebar (three controls only) ----------
# We use direct sidebar inputs (no form) so inputs are instantly in session_state and masked.
groq_api_key = st.sidebar.text_input("Groq API Key", type="password", key="groq_api_key")
gemini_api_key = st.sidebar.text_input("Gemini (Google) API Key", type="password", key="gemini_api_key")

# Groq model dropdown with the models you requested (adjust to exact model names from your account)
groq_model = st.sidebar.selectbox(
    "Groq model",
    options=[
        "llama-3.3-70b-versatile",
        "llama-3.1-8b-instant",
        "chat.groq"
    ],
    index=0,
    key="groq_model"
)

# Fixed constant: embedding model (hidden from UI)
EMBEDDING_MODEL = "gemini-embedding-001"

# Fixed top_k
TOP_K = 4

# Warnings if optional dependencies are missing
if not GOOGLE_GENAI_AVAILABLE:
    st.sidebar.error("Missing langchain-google-genai package — Gemini embeddings will not work until installed.")
if Chroma is None:
    st.sidebar.error("Missing Chroma wrapper (langchain_community). Install required packages.")
if not GROQ_SDK_AVAILABLE:
    st.sidebar.info("Groq SDK not installed; the app will fallback to HTTP requests for Groq calls.")

# ---------- Helpers ----------
def show_white_error(msg: str):
    """Show an error message styled to remain white on violet background."""
    st.markdown(f'<div style="color: white; background: rgba(0,0,0,0.18); padding:10px; border-radius:6px;">'
                f'<strong>Error:</strong> {msg}</div>', unsafe_allow_html=True)

def show_white_info(msg: str):
    st.markdown(f'<div style="color: white; padding:8px;">{msg}</div>', unsafe_allow_html=True)

def test_gemini_key(embeddings, sample_text="hello"):
    """
    Run a tiny embedding request to validate the provided Gemini API key.
    Raises Exception on failure.
    """
    # Many embeddings wrappers expose embed_documents or embed_query — try common ones.
    if hasattr(embeddings, "embed_documents"):
        return embeddings.embed_documents([sample_text])
    if hasattr(embeddings, "embed_queries"):
        return embeddings.embed_queries([sample_text])
    if hasattr(embeddings, "embed_query"):
        return embeddings.embed_query(sample_text)
    raise RuntimeError("Embeddings wrapper has no known embed method.")

# ---------- File uploader and pipeline ----------
uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

vectorstore = None
texts = []

if uploaded_file:
    status = st.empty()
    status.info("Uploading PDF...")

    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    status.info("Extracting text from PDF...")
    try:
        reader = PdfReader(tmp_path)
        raw_text = ""
        for p in reader.pages:
            txt = p.extract_text()
            if txt:
                raw_text += txt + "\n"
    except Exception as e:
        show_white_error(f"Failed to read PDF: {e}")
        st.stop()

    if not raw_text.strip():
        show_white_error("No extractable text found in the uploaded PDF.")
        st.stop()

    status.info("Splitting text into chunks...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""],
    )
    texts = splitter.split_text(raw_text)
    status.info(f"{len(texts)} chunks created — preparing embeddings...")

    # Validate Gemini key presence
    if not gemini_api_key:
        show_white_error("Gemini (Google) API key is required. Paste it in the sidebar (it will appear masked).")
        st.stop()

    # Validate embedding package presence
    if not GOOGLE_GENAI_AVAILABLE or GoogleGenerativeAIEmbeddings is None:
        show_white_error("Gemini embeddings package 'langchain-google-genai' is not available in this environment.")
        st.stop()

    # Initialize embeddings and do a small test embed to validate the API key.
    try:
        embeddings = GoogleGenerativeAIEmbeddings(google_api_key=gemini_api_key, model=EMBEDDING_MODEL)
    except Exception as e:
        show_white_error(f"Failed to initialize Gemini embeddings wrapper: {e}")
        st.stop()

    # Test the key with a tiny call and handle errors gracefully (user saw earlier INVALID_ARGUMENT).
    try:
        # small test to surface invalid API key right away
        _ = test_gemini_key(embeddings, sample_text="validation test")
    except Exception as e:
        # parse some common error shapes to present actionable advice
        err_text = str(e)
        if "API key not valid" in err_text or "API_KEY_INVALID" in err_text or "401" in err_text:
            show_white_error(
                "Gemini API key validation failed: the API key appears invalid. "
                "Please confirm you pasted the correct key and that the key has access to the Generative Language API (Vertex/Gemini)."
            )
        else:
            show_white_error(f"Gemini embedding test failed: {err_text}")
        st.stop()

    # Build Chroma vectorstore
    if Chroma is None:
        show_white_error("Chroma vectorstore wrapper is not available in this environment. Install required package.")
        st.stop()

    try:
        # Many Chroma wrappers accept parameter name 'embedding' or 'embeddings'
        # We'll attempt a widely used signature.
        vectorstore = Chroma.from_texts(texts=texts, embedding=embeddings)
    except Exception as e:
        show_white_error(f"Failed to create vectorstore: {e}")
        st.stop()

    status.success("Document embedded successfully and ready to query.")
    st.markdown("---")
    st.subheader("Ask questions about the uploaded PDF")
    question = st.text_input("Enter your question:")

    if st.button("Get answer") and question.strip():
        # Validate Groq key
        if not groq_api_key:
            show_white_error("Groq API key is required. Enter it in the sidebar (masked).")
            st.stop()

        # Retrieve top-k docs. Try both APIs (similarity_search_with_score preferred)
        try:
            if hasattr(vectorstore, "similarity_search_with_score"):
                hits = vectorstore.similarity_search_with_score(question, k=TOP_K)
                docs = [d for d, _ in hits]
                scores = [s for _, s in hits]
            elif hasattr(vectorstore, "similarity_search"):
                docs = vectorstore.similarity_search(question, k=TOP_K)
                scores = None
            else:
                raise RuntimeError("Vectorstore retrieval method not found.")
        except Exception as e:
            show_white_error(f"Document retrieval failed: {e}")
            st.stop()

        # Build context from retrieved docs (each snippet truncated)
        def safe_snip(t: str, n: int = 2000) -> str:
            return t.replace("\n", " ")[:n]

        context_parts = [safe_snip(getattr(d, "page_content", str(d))) for d in docs]
        context_text = "\n\n---\n\n".join(context_parts)

        prompt = (
            "You are a helpful assistant that answers questions using the provided document context. "
            "If the answer cannot be found in the context, reply: \"I don't know based on the provided document.\""
            f"\n\nContext:\n{context_text}\n\nQuestion: {question}\n\nAnswer:"
        )

        show_white_info("Querying Groq LLM now...")

        # Attempt Groq SDK first
        answer_text: Optional[str] = None
        groq_error: Optional[str] = None

        if GROQ_SDK_AVAILABLE and Groq is not None:
            try:
                client = Groq(api_key=groq_api_key)
                # Try chat-like method
                try:
                    resp = client.chat.create(messages=[{"role": "user", "content": prompt}], model=groq_model)
                    # Many SDK responses are dict-like; try to extract content
                    if isinstance(resp, dict):
                        if resp.get("content"):
                            answer_text = resp["content"]
                        elif "choices" in resp and resp["choices"]:
                            ch = resp["choices"][0]
                            if isinstance(ch, dict) and "message" in ch and "content" in ch["message"]:
                                answer_text = ch["message"]["content"]
                            elif "text" in ch:
                                answer_text = ch["text"]
                            else:
                                answer_text = str(ch)
                        else:
                            answer_text = str(resp)
                    else:
                        answer_text = str(resp)
                except Exception:
                    # Try alternative SDK endpoint shape
                    try:
                        resp2 = client.completions.create(prompt=prompt, model=groq_model)
                        if isinstance(resp2, dict) and "choices" in resp2:
                            answer_text = resp2["choices"][0].get("text", "")
                        else:
                            answer_text = str(resp2)
                    except Exception as e:
                        groq_error = f"Groq SDK call failed: {e}"
            except Exception as e:
                groq_error = f"Groq SDK initialization failed: {e}"

        # If no SDK answer, attempt HTTP fallback
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
                    "temperature": 0.0
                }
                BASE_URL = "https://api.groq.ai/v1"  # adjust if Groq provides a different base URL
                endpoint = f"{BASE_URL}/chat/completions"
                resp = requests.post(endpoint, headers=headers, json=payload, timeout=30)
                if resp.status_code == 200:
                    j = resp.json()
                    if "choices" in j and j["choices"]:
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
                    groq_error = f"Groq HTTP error {resp.status_code}: {resp.text}"
            except Exception as e:
                groq_error = f"Groq HTTP fallback exception: {e}"

        if not answer_text:
            show_white_error("Failed to obtain an answer from Groq.")
            if groq_error:
                st.markdown(f'<div style="color:white">Details: {groq_error}</div>', unsafe_allow_html=True)
            st.stop()

        # Present answer and top documents (all text forced white by CSS)
        st.markdown("**ANSWER**")
        st.markdown(f'<div style="color:white; background: rgba(0,0,0,0.12); padding:12px; border-radius:6px;">{answer_text}</div>', unsafe_allow_html=True)

        st.markdown("**Top documents (by relevance)**")
        if scores:
            for d, s in zip(docs, scores):
                snippet = getattr(d, "page_content", str(d))[:700].replace("\n", " ")
                st.markdown(f'<div style="color:white">[{s:.4f}] {snippet}...</div>', unsafe_allow_html=True)
        else:
            for i, d in enumerate(docs, start=1):
                snippet = getattr(d, "page_content", str(d))[:700].replace("\n", " ")
                st.markdown(f'<div style="color:white">[{i}] {snippet}...</div>', unsafe_allow_html=True)

# Footer note
st.markdown("---")
st.markdown(
    '<div style="color:white">Notes: '
    'If you get "Gemini API key invalid" please verify the key is correct, has access to the Generative Language API '
    '(Vertex AI/Gemini) and is not restricted by IP/Referrer. For Groq, make sure the model name in the dropdown matches '
    'your Groq account. If Groq uses a different base URL for your tenant, edit BASE_URL accordingly in the code.</div>',
    unsafe_allow_html=True
)
