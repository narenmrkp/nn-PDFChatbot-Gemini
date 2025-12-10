import os
import tempfile
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain.vectorstores import Chroma

# Google + LangChain embeddings integration
try:
    from langchain_google_genai import GoogleGenerativeAIEmbeddings
    GOOGLE_GENAI_AVAILABLE = True
except Exception:
    GOOGLE_GENAI_AVAILABLE = False

# Groq client (official Python SDK)
try:
    from groq import Groq
    GROQ_AVAILABLE = True
except Exception:
    GROQ_AVAILABLE = False

st.set_page_config(page_title="NN's PDF Q&A Chatbot", layout='wide')

# --- Styles: radiant blue-violet background, white text
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(135deg, #3a0ca3 0%, #7209b7 50%, #4b0082 100%);
        color: white;
    }
    .stTextInput>div>div>input { background-color: rgba(255,255,255,0.06); color: white; }
    .stButton>button { background-color: rgba(255,255,255,0.08); color: white; }
    .stSidebar .sidebar-content { background: transparent; }
    h1 {color: white}
    </style>
    """ ,
    unsafe_allow_html=True,
)

st.title("NN's PDF Q&A Chatbot")
st.write("Upload a PDF, embed it with Gemini embeddings, and ask questions. Choose Groq as the LLM provider.")

# Sidebar: provider selection and keys (no .env needed)
with st.sidebar.form(key='settings'):
    st.header("Keys & Provider")
    provider = st.radio("Choose LLM provider", ("Groq",))
    groq_api_key = st.text_input("Groq API Key (enter when you run the app)", type="password")
    # Google key for embeddings
    google_api_key = st.text_input("Google (Gemini/Vertex AI) API Key", type="password")
    submitted = st.form_submit_button("Save settings")

if submitted:
    st.session_state['groq_api_key'] = groq_api_key
    st.session_state['google_api_key'] = google_api_key
    st.success("Settings saved in session — keys are used only at runtime.")

# Use session values if available
groq_api_key = st.session_state.get('groq_api_key', '')
google_api_key = st.session_state.get('google_api_key', '')

# Check dependencies availability
if not GOOGLE_GENAI_AVAILABLE:
    st.sidebar.error("Dependency missing: install 'langchain-google-genai' to enable Gemini embeddings.")
if not GROQ_AVAILABLE:
    st.sidebar.error("Dependency missing: install 'groq' to enable Groq LLM integration.")

# File uploader
uploaded_file = st.file_uploader("Upload a PDF file", type=['pdf'])

vectorstore = None
texts = []

if uploaded_file is not None:
    status = st.empty()
    status.info("Document uploading...")

    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    status.info("PDF uploaded successfully — extracting text...")

    reader = PdfReader(tmp_path)
    raw_text = ""
    for p in reader.pages:
        text = p.extract_text()
        if text:
            raw_text += text + "\n"

    if not raw_text.strip():
        st.error("No extractable text found in PDF.")
    else:
        status.info("Text extracted — splitting into chunks...")

        text_splitter = CharacterTextSplitter(separator="\n", chunk_size=800, chunk_overlap=200, length_function=len)
        texts = text_splitter.split_text(raw_text)

        status.info(f"{len(texts)} chunks created — embedding (Gemini) ...")

        if not google_api_key:
            st.error("Google API key is required for Gemini embeddings. Enter it in the sidebar and re-upload or re-embed.")
            st.stop()

        # Instantiate Gemini embeddings via LangChain's GoogleGenerativeAIEmbeddings
        try:
            embeddings = GoogleGenerativeAIEmbeddings(google_api_key=google_api_key, model="gemini-embedding-001")
        except Exception as e:
            st.error(f"Failed to initialize Gemini embeddings: {e}")
            st.stop()

        # Build a Chroma vectorstore locally (fast & simple)
        try:
            vectorstore = Chroma.from_texts(texts, embedding=embeddings)
        except Exception as e:
            st.error(f"Failed to create Chroma vectorstore: {e}")
            st.stop()

        index = VectorStoreIndexWrapper(vectorstore=vectorstore)
        status.success("Document embedded successfully and ready to query.")

        st.markdown("---")
        st.subheader("Ask questions")
        question = st.text_input("Enter your question:")

        if st.button("Get answer") and question.strip():
            if not groq_api_key:
                st.error("Groq API key is required. Enter it in the sidebar.")
                st.stop()

            st.info(f"QUESTION: {question}")

            # Build a simple Groq LLM wrapper using groq client (OpenAI-compatible endpoints supported)
            try:
                client = Groq(api_key=groq_api_key)
            except Exception as e:
                st.error(f"Failed to initialize Groq client: {e}")
                client = None

            # Use the index to retrieve context and then call Groq for final answer
            with st.spinner("Retrieving context and querying Groq..."):
                try:
                    # Retrieve top docs
                    docs = vectorstore.similarity_search_with_score(question, k=4)
                    context_text = "\n\n".join([d.page_content for d, _ in docs])

                    # Compose prompt for Groq
                    prompt = (
                        "You are a helpful assistant. Use the provided context from a PDF to answer the question. "
                        "If the answer is not found in the context, say you don't know.\n\n"
                        f"Context:\n{context_text}\n\nQuestion: {question}\nAnswer:"
                    )

                    # Call Groq Chat/Completion endpoint compatible call
                    # The Groq Python client offers a chat/completion API similar to OpenAI's—adjust model name if needed.
                    response = client.chat.create(messages=[{"role": "user", "content": prompt}], model="chat.groq")

                    # response parsing depends on Groq SDK shape
                    answer_text = response.get('content') if isinstance(response, dict) else str(response)

                except Exception as e:
                    st.error(f"Failed to query index or Groq: {e}")
                    answer_text = None

            if answer_text:
                st.success("ANSWER")
                st.write(answer_text)

                st.markdown("**Top documents (by relevance)**")
                for doc, score in docs:
                    st.write(f"[{score:.4f}] {doc.page_content[:500].replace('\n', ' ')}...")

# Footer
st.markdown("---")
st.caption("Notes: This app uses Gemini embeddings via LangChain's GoogleGenerativeAIEmbeddings and Groq as the LLM. Keys are entered at runtime in the sidebar. If you need persistent vector storage (Astra, Milvus, Pinecone), I can add that." )
