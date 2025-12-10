"""
NN's PDF Q&A Chatbot — Modern GenAI UI
- "Deep Night" Aesthetic: Dark gradients, glassmorphism, and neon accents.
- Improved UX: Clean Sidebar, Expander for Keys, polished typography.
"""

import os
import tempfile
import json
from typing import List, Optional
import requests
import streamlit as st
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Optional Imports with Graceful Fallbacks
try:
    from langchain_community.vectorstores import Chroma
except ImportError:
    Chroma = None

try:
    from langchain_google_genai import GoogleGenerativeAIEmbeddings
    GOOGLE_GENAI_AVAILABLE = True
except ImportError:
    GoogleGenerativeAIEmbeddings = None
    GOOGLE_GENAI_AVAILABLE = False

try:
    from groq import Groq
    GROQ_SDK_AVAILABLE = True
except ImportError:
    Groq = None
    GROQ_SDK_AVAILABLE = False

# -----------------------------------------------------------------------------
# 1. Page Configuration & Custom CSS (The "Look")
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="GenAI PDF Explorer",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Modern "Deep Night" CSS Theme
st.markdown(
    """
    <style>
    /* 1. MAIN BACKGROUND: Deep Gradient */
    .stApp {
        background: radial-gradient(circle at 10% 20%, #0f172a 0%, #1e1e2e 40%, #000000 100%);
        color: #e2e8f0; /* Light gray text for readability */
        font-family: 'Inter', sans-serif;
    }

    /* 2. SIDEBAR: Glassmorphism */
    section[data-testid="stSidebar"] {
        background-color: rgba(23, 23, 35, 0.6);
        backdrop-filter: blur(12px);
        border-right: 1px solid rgba(255, 255, 255, 0.05);
    }
    
    /* 3. INPUTS & WIDGETS */
    .stTextInput > div > div > input, 
    .stSelectbox > div > div > div {
        background-color: rgba(255, 255, 255, 0.05) !important;
        color: #f1f5f9 !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 8px !important;
    }
    .stTextInput > div > div > input:focus {
        border-color: #38bdf8 !important; /* Sky blue focus */
        box-shadow: 0 0 10px rgba(56, 189, 248, 0.2);
    }

    /* 4. BUTTONS: Gradient Primary */
    .stButton > button {
        background: linear-gradient(90deg, #4f46e5 0%, #8b5cf6 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 0.6rem 1.2rem !important;
        font-weight: 600 !important;
        transition: opacity 0.2s ease-in-out;
    }
    .stButton > button:hover {
        opacity: 0.9;
        box-shadow: 0 4px 12px rgba(99, 102, 241, 0.4);
    }

    /* 5. TEXT STYLING */
    h1, h2, h3 {
        color: #f8fafc !important; 
        font-weight: 700 !important;
    }
    h1 {
        background: linear-gradient(90deg, #38bdf8, #818cf8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        padding-bottom: 0.5rem;
    }
    .stMarkdown p, .stText {
        font-size: 1.05rem;
        line-height: 1.6;
    }
    
    /* 6. CONTAINERS (Answer Box) */
    .answer-box {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-left: 4px solid #38bdf8;
        border-radius: 8px;
        padding: 20px;
        margin-top: 20px;
    }
    
    /* Remove top margin clutter */
    .block-container {
        padding-top: 2rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# 2. Sidebar: Configuration & Keys
# -----------------------------------------------------------------------------
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/9626/9626620.png", width=60) # Generic Bot Icon
    st.title("Settings")
    
    with st.expander("🔑 API Keys", expanded=True):
        st.markdown("Required for embedding and chatting.")
        groq_api_key = st.text_input("Groq API Key", type="password", help="Enter your Groq Cloud API Key")
        gemini_api_key = st.text_input("Gemini API Key", type="password", help="Enter your Google Gemini API Key")
        
    with st.expander("⚙️ Model Options", expanded=False):
        groq_model = st.selectbox(
            "Select LLM",
            options=[
                "llama-3.3-70b-versatile",
                "llama-3.1-8b-instant",
                "mixtral-8x7b-32768"
            ],
            index=0
        )
        st.caption("Using `gemini-embedding-001` for vector embeddings.")

    st.markdown("---")
    
    # Status Indicators
    if not GROQ_SDK_AVAILABLE:
        st.warning("⚠️ Groq SDK not found. Using HTTP fallback.")
    if not GOOGLE_GENAI_AVAILABLE:
        st.error("❌ `langchain-google-genai` missing.")

# -----------------------------------------------------------------------------
# 3. Main Interface
# -----------------------------------------------------------------------------

col1, col2 = st.columns([2, 1])
with col1:
    st.title("GenAI PDF Explorer")
    st.markdown("Upload a document, ask questions, and get AI-powered insights immediately.")

# Helper: Constants
EMBEDDING_MODEL = "gemini-embedding-001"
TOP_K = 4

# Helper: State Management
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "file_processed" not in st.session_state:
    st.session_state.file_processed = False

# -----------------------------------------------------------------------------
# 4. File Processing Logic
# -----------------------------------------------------------------------------

# File Uploader in a nice container
upload_container = st.container()
with upload_container:
    uploaded_file = st.file_uploader("📂 Upload PDF Document", type=["pdf"])

if uploaded_file and not st.session_state.file_processed:
    # Check keys first
    if not gemini_api_key:
        st.info("👈 Please enter your **Gemini API Key** in the sidebar to process the file.")
        st.stop()
        
    if not GOOGLE_GENAI_AVAILABLE or GoogleGenerativeAIEmbeddings is None:
        st.error("Missing required packages for embeddings. Please check `requirements.txt`.")
        st.stop()

    with st.status("🚀 Processing Document...", expanded=True) as status:
        try:
            # 1. Extract Text
            status.write("Reading PDF...")
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name
            
            reader = PdfReader(tmp_path)
            raw_text = ""
            for p in reader.pages:
                txt = p.extract_text()
                if txt: raw_text += txt + "\n"
            
            if not raw_text.strip():
                status.update(label="Failed!", state="error")
                st.error("No text could be extracted from this PDF.")
                st.stop()
                
            # 2. Split Text
            status.write("Splitting content...")
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            texts = splitter.split_text(raw_text)
            
            # 3. Embed
            status.write("Generating Embeddings (Gemini)...")
            embeddings = GoogleGenerativeAIEmbeddings(google_api_key=gemini_api_key, model=EMBEDDING_MODEL)
            
            # Simple test to fail fast if key is bad
            try:
                embeddings.embed_query("test")
            except Exception as e:
                status.update(label="Authentication Failed", state="error")
                st.error(f"Gemini API Error: {str(e)}")
                st.stop()

            # 4. Store
            if Chroma is None:
                raise ImportError("ChromaDB is not installed.")
                
            status.write("Indexing into Vector Store...")
            st.session_state.vectorstore = Chroma.from_texts(texts=texts, embedding=embeddings)
            st.session_state.file_processed = True
            
            status.update(label="✅ Ready to Chat!", state="complete", expanded=False)
            
        except Exception as e:
            status.update(label="Error Occurred", state="error")
            st.error(f"Processing Error: {str(e)}")
            st.stop()

# -----------------------------------------------------------------------------
# 5. Q&A Interface
# -----------------------------------------------------------------------------

if st.session_state.file_processed:
    st.divider()
    
    # Input Area
    prompt = st.chat_input("Ask a question about your document...")
    
    if prompt:
        # Display User Message
        with st.chat_message("user"):
            st.write(prompt)
            
        if not groq_api_key:
            st.error("Please enter your **Groq API Key** in the sidebar.")
            st.stop()

        # Processing Message
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            message_placeholder.markdown("🔍 *Scanning document...*")
            
            # 1. Retrieve Context
            try:
                vectorstore = st.session_state.vectorstore
                # Search
                hits = vectorstore.similarity_search_with_score(prompt, k=TOP_K)
                docs = [d for d, _ in hits]
                
                # Format Context
                context_text = "\n\n".join([d.page_content for d in docs])
                
            except Exception as e:
                st.error(f"Retrieval Error: {e}")
                st.stop()

            # 2. Construct Prompt
            full_prompt = (
                "You are an intelligent assistant analyzing a provided document. "
                "Use the context below to answer the user's question accurately and concisely. "
                "If the answer is not in the context, say you don't know.\n\n"
                f"CONTEXT:\n{context_text}\n\n"
                f"USER QUESTION: {prompt}"
            )

            # 3. Call LLM (Groq)
            message_placeholder.markdown("🤔 *Thinking...*")
            answer_text = ""
            
            try:
                if GROQ_SDK_AVAILABLE and Groq:
                    # SDK Method
                    client = Groq(api_key=groq_api_key)
                    completion = client.chat.completions.create(
                        messages=[{"role": "user", "content": full_prompt}],
                        model=groq_model,
                        temperature=0.1
                    )
                    answer_text = completion.choices[0].message.content
                else:
                    # HTTP Fallback
                    headers = {
                        "Authorization": f"Bearer {groq_api_key}",
                        "Content-Type": "application/json"
                    }
                    payload = {
                        "model": groq_model,
                        "messages": [{"role": "user", "content": full_prompt}],
                        "temperature": 0.1
                    }
                    resp = requests.post("https://api.groq.ai/v1/chat/completions", headers=headers, json=payload)
                    if resp.status_code == 200:
                        data = resp.json()
                        answer_text = data['choices'][0]['message']['content']
                    else:
                        raise Exception(f"Groq API {resp.status_code}: {resp.text}")

            except Exception as e:
                st.error(f"LLM Error: {e}")
                st.stop()

            # 4. Final Display
            message_placeholder.markdown(answer_text)
            
            # Optional: Show functionality to see sources
            with st.expander("📚 View Source Snippets"):
                for i, doc in enumerate(docs):
                    st.markdown(f"**Source {i+1}:**")
                    st.caption(doc.page_content[:400] + "...")

elif not uploaded_file:
    # Empty State - Hero Section feel
    st.info("👆 Start by uploading a PDF document above.")
