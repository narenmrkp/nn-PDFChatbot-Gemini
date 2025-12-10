"""
NN's PDF Q&A Chatbot — UI fixes for masked keys and sidebar color
- Sidebar fields are masked (Streamlit's type="password")
- Additional visible masked indicator (fixed '********') shown when a key exists
- Sidebar background set to vibrant golden-orange gradient (CSS)
- All app text forced to white for readability on violet background
- Embeddings: Gemini (gemini-embedding-001 hard-coded)
- LLM: Groq (SDK preferred, otherwise HTTP fallback)
- Vectorstore: Chroma local (langchain_community wrapper)
"""

import os
import tempfile
import json
from typing import List, Optional

import streamlit as st
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Chroma wrapper (langchain_community)
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

# Streamlit page config
st.set_page_config(page_title="NN's PDF Q&A Chatbot", layout="wide")

# ----------------------
# CSS: violet background, white text, golden-orange sidebar
# ----------------------
st.markdown(
    """
    <style>
    /* App background */
    .stApp {
        background: linear-gradient(135deg, #3a0ca3 0%, #7209b7 50%, #4b0082 100%) !important;
        color: white !important;
    }

    /* Sidebar styling: vibrant golden-orange gradient */
    /* Two selectors to increase compatibility across Streamlit versions */
    .stSidebar .sidebar-content, .css-1d391kg .stSidebar {
        background: linear-gradient(180deg, #ffb703 0%, #fb8500 100%) !important;
        color: white !important;
        padding: 16px !important;
        border-radius: 8px !important;
    }

    /* Force text color white in input, markdown, and outputs */
    .stTextInput, .stTextArea, .stMarkdown, .stButton, .stExpander, .stMetric {
        color: white !important;
    }
    input, textarea {
        color: white !important;
        background-color: rgba(255,255,255,0.06) !important;
    }

    /* Force alert/exception/error/warning to white text */
    .stAlert, .stException, .stError, .stWarning {
        color: white !important;
        background: rgba(0,0,0,0.18) !important;
    }

    /* Ensure outputs and captions remain white */
    .streamlit-expanderHeader, .stMarkdown, .stText, .stCaption {
        color: white !important;
    }

    /* Buttons */
    .stButton>button {
        background-color: rgba(255,255,255,0.08) !important;
        color: white !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Title and description
st.title("NN's PDF Q&A Chatbot")
st.write("Upload a PDF, embed with Gemini embeddings, and query using Groq. Enter keys in the sidebar (masked).")

# ----------------------
# Sidebar: three controls only (masked inputs + model dropdown)
# ----------------------
# Use direct sidebar inputs (type="password" ensures characters typed are masked)
groq_api_key_input = st.sidebar.text_input("Groq API Key", type="password", key="groq_api_key_input")
gemini_api_key_input = st.sidebar.text_input("Gemini (Google) API Key", type="password", key="gemini_api_key_input")

# Groq model dropdown (example models you requested)
groq_model = st.sidebar.selectbox(
    "Groq model",
    options=[
        "llama-3.3-70b-versatile",
        "llama-3.1-8b-instant",
        "chat.groq"
    ],
    index=0,
    key="groq_model_select"
)

# Show masked confirmation right below each input (fixed stars so the real key isn't revealed)
if groq_api_key_input:
    st.sidebar.markdown("**Groq key:** `********`")
else:
    st.sidebar.markdown("**Groq key:** _not set_")

if gemini_api_key_input:
    st.sidebar.markdown("**Gemini key:** `********`")
else:
    st.sidebar.markdown("**Gemini key:** _not set_")

# constants
EMBEDDING_MODEL = "gemini-embedding-001"
TOP_K = 4

# Warnings for missing packages
if not GOOGLE_GENAI_AVAILABLE:
    st.sidebar.error("Missing 'langchain-google-genai' — Gemini embeddings unavailable until installed.")
if Chroma is None:
    st.sidebar.error("Missing Chroma wrapper (langchain_community). Install required package.")
if not GROQ_SDK_AVAILABLE:
    st.sidebar.info("Groq SDK not installed; app will attempt HTTP fallback for Groq calls.")

# Helper functions to display white-styled messages
def white_error(msg: str):
    st.markdown(f'<div style="color:white; background: rgba(0,0,0,0.18); padding:10px; border-radius:6px;"><strong>Error:</strong> {msg}</div>', unsafe_allow_html=True)

def white_info(msg: str):
    st.markdown(f'<div style="color:white; padding:8px;">{msg}</div>', unsafe_allow_html=True)

# small embedding test function
def test_embedding_wrapper(emb, sample="validation test"):
    if hasattr(emb, "embed_documents"):
        return emb.embed_documents([sample])
    if hasattr(emb, "embed_queries"):
        return emb.embed_queries([sample])
    if hasattr(emb, "embed_query"):
        return emb.embed_query(sample)
    raise RuntimeError("Embeddings wrapper has no known embed method.")

# ----------------------
# File uploader and processing
# ----------------------
uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])
vectorstore = None
texts: List[str] = []

if uploaded_file:
    status = st.empty()
    status.info("Uploading PDF...")

    # Save to tempfile
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
        white_error(f"Failed to read PDF: {e}")
        st.stop()

    if not raw_text.strip():
        white_error("No extractable text found in the uploaded PDF.")
        st.stop()

    status.info("Splitting text into chunks...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=200, separators=["\n\n", "\n", " ", ""])
    texts = splitter.split_text(raw_text)
    status.info(f"{len(texts)} chunks created — preparing embeddings...")

    # Validate Gemini key input presence
    if not gemini_api_key_input:
        white_error("Gemini (Google) API key is required. Paste it in the sidebar (it will appear masked).")
        st.stop()

    # Validate embedding package
    if not GOOGLE_GENAI_AVAILABLE or GoogleGenerativeAIEmbeddings is None:
        white_error("Gemini embeddings package 'langchain-google-genai' is not installed in this environment.")
        st.stop()

    # Initialize embeddings with the provided key (this wrapper expects a google_api_key argument)
    try:
        embeddings = GoogleGenerativeAIEmbeddings(google_api_key=gemini_api_key_input, model=EMBEDDING_MODEL)
    except Exception as e:
        # show user-friendly guidance
        err = str(e)
        if "API key not valid" in err or "API_KEY_INVALID" in err or "401" in err:
            white_error("Gemini API key validation failed: the API key looks invalid. Verify the key and that Generative Language API is enabled.")
        else:
            white_error(f"Failed to initialize Gemini embeddings: {err}")
        st.stop()

    # Test the key via a small embedding call to surface invalid-key errors early
    try:
        _ = test_embedding_wrapper(embeddings)
    except Exception as e:
        err = str(e)
        if "API key not valid" in err or "API_KEY_INVALID" in err or "401" in err:
            white_error("Gemini API key validation failed during test embed: the key appears invalid or lacks permission. Check the Google Cloud Console.")
        else:
            white_error(f"Gemini embedding test failed: {err}")
        st.stop()

    # Build Chroma vectorstore
    if Chroma is None:
        white_error("Chroma vectorstore wrapper is not available. Install 'langchain_community' or the appropriate package.")
        st.stop()

    try:
        vectorstore = Chroma.from_texts(texts=texts, embedding=embeddings)
    except Exception as e:
        white_error(f"Failed to create vectorstore: {e}")
        st.stop()

    status.success("Document embedded successfully and ready to query.")
    st.markdown("---")
    st.subheader("Ask questions about the uploaded PDF")
    question = st.text_input("Enter your question:")

    if st.button("Get answer") and question.strip():
        if not groq_api_key_input:
            white_error("Groq API key is required. Enter it in the sidebar (masked).")
            st.stop()

        # Retrieve top documents
        try:
            if hasattr(vectorstore, "similarity_search_with_score"):
                hits = vectorstore.similarity_search_with_score(question, k=TOP_K)
                docs = [d for d, _ in hits]
                scores = [s for _, s in hits]
            elif hasattr(vectorstore, "similarity_search"):
                docs = vectorstore.similarity_search(question, k=TOP_K)
                scores = None
            else:
                raise RuntimeError("Vectorstore retrieval API not found.")
        except Exception as e:
            white_error(f"Retrieval failed: {e}")
            st.stop()

        def safe_snip(t: str, n: int = 2000) -> str:
            return t.replace("\n", " ")[:n]

        context_parts = [safe_snip(getattr(d, "page_content", str(d))) for d in docs]
        context_text = "\n\n---\n\n".join(context_parts)

        prompt = (
            "You are a helpful assistant that answers questions using the provided PDF context. "
            "If the answer cannot be found in the context, reply: \"I don't know based on the provided document.\""
            f"\n\nContext:\n{context_text}\n\nQuestion: {question}\n\nAnswer:"
        )

        white_info("Querying Groq LLM for the answer...")

        answer_text: Optional[str] = None
        groq_error: Optional[str] = None

        # Try Groq SDK first
        if GROQ_SDK_AVAILABLE and Groq is not None:
            try:
                client = Groq(api_key=groq_api_key_input)
                try:
                    resp = client.chat.create(messages=[{"role": "user", "content": prompt}], model=groq_model)
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

        # HTTP fallback for Groq
        if not answer_text:
            try:
                headers = {
                    "Authorization": f"Bearer {groq_api_key_input}",
                    "Content-Type": "application/json",
                }
                payload = {
                    "model": groq_model,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 800,
                    "temperature": 0.0,
                }
                BASE_URL = "https://api.groq.ai/v1"
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
            white_error("Failed to obtain an answer from Groq.")
            if groq_error:
                st.markdown(f'<div style="color:white">Details: {groq_error}</div>', unsafe_allow_html=True)
            st.stop()

        # Present answer (white text)
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

# Footer
st.markdown("---")
st.markdown(
    '<div style="color:white">Notes: The input fields are masked using Streamlit\'s password input. '
    'Because Streamlit does not allow customizing the in-field mask character, this app shows a fixed `********` '
    'indicator beneath each input when a key is present. If Gemini key returns `API key not valid`, verify the key '
    'and ensure the Generative Language API is enabled in Google Cloud. If Groq calls fail, confirm the exact model name '
    'and update BASE_URL if your Groq tenant uses a different endpoint.</div>',
    unsafe_allow_html=True
)
