"""
PDF QA Bot — Streamlit Web UI.
Built by Totan
"""

import os
import tempfile

import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

# ---------------- Load API key ----------------
load_dotenv()

def get_groq_api_key() -> str:
    try:
        key = st.secrets["GROQ_API_KEY"]
        if key:
            return key
    except Exception:
        pass
    key = os.getenv("GROQ_API_KEY", "")
    if not key:
        st.error("❌ GROQ_API_KEY not found. Add it to Streamlit secrets or a local .env file.")
        st.stop()
    return key

GROQ_API_KEY = get_groq_api_key()

# ---------------- Page config ----------------
st.set_page_config(page_title="PDF QA Bot", page_icon="📄", layout="centered")

# ---------------- Custom CSS ----------------
st.markdown("""
<style>
    /* Import Google Font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Animated gradient background on main header */
    .hero-banner {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f64f59 100%);
        background-size: 200% 200%;
        animation: gradientShift 6s ease infinite;
        border-radius: 16px;
        padding: 2.5rem 2rem;
        text-align: center;
        margin-bottom: 1.5rem;
        box-shadow: 0 8px 32px rgba(102, 126, 234, 0.4);
    }

    @keyframes gradientShift {
        0%   { background-position: 0% 50%; }
        50%  { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    .hero-title {
        font-size: 2.6rem;
        font-weight: 700;
        color: white;
        margin: 0;
        letter-spacing: -0.5px;
        text-shadow: 0 2px 8px rgba(0,0,0,0.2);
    }

    .hero-subtitle {
        font-size: 1rem;
        color: rgba(255,255,255,0.85);
        margin-top: 0.5rem;
        font-weight: 300;
    }

    /* Pulse animation on the icon */
    .pulse-icon {
        display: inline-block;
        animation: pulse 2s ease-in-out infinite;
        font-size: 3rem;
    }

    @keyframes pulse {
        0%, 100% { transform: scale(1); }
        50%       { transform: scale(1.12); }
    }

    /* Answer card */
    .answer-card {
        background: linear-gradient(135deg, #f0fff4, #e6fffa);
        border-left: 4px solid #38a169;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        margin-top: 1rem;
        animation: fadeInUp 0.5s ease;
        box-shadow: 0 4px 16px rgba(56, 161, 105, 0.15);
    }

    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(16px); }
        to   { opacity: 1; transform: translateY(0); }
    }

    .answer-label {
        font-weight: 700;
        color: #276749;
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 0.5rem;
    }

    .answer-text {
        color: #1a202c;
        font-size: 1.05rem;
        line-height: 1.7;
    }

    /* Upload area styling */
    .upload-section {
        background: #f7fafc;
        border: 2px dashed #cbd5e0;
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        transition: border-color 0.3s;
        margin-bottom: 1rem;
    }

    /* Stats chips */
    .chip {
        display: inline-block;
        background: #ebf4ff;
        color: #2b6cb0;
        border-radius: 20px;
        padding: 0.25rem 0.75rem;
        font-size: 0.8rem;
        font-weight: 600;
        margin: 0.2rem;
    }

    /* Builder badge at bottom */
    .builder-badge {
        text-align: center;
        margin-top: 3rem;
        padding: 1rem;
        background: linear-gradient(135deg, #667eea15, #764ba215);
        border-radius: 12px;
        border: 1px solid #667eea30;
        animation: fadeInUp 1s ease;
    }

    .builder-badge p {
        margin: 0;
        font-size: 0.9rem;
        color: #553c9a;
        font-weight: 500;
    }

    .builder-name {
        font-size: 1.1rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea, #764ba2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }

    /* Sidebar styling */
    .sidebar-card {
        background: white;
        border-radius: 10px;
        padding: 0.8rem;
        margin-bottom: 0.5rem;
        border: 1px solid #e2e8f0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }

    /* Hide default streamlit header decorations */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ---------------- Hero Banner ----------------
st.markdown("""
<div class="hero-banner">
    <div class="pulse-icon">📄</div>
    <h1 class="hero-title">PDF QA Bot</h1>
    <p class="hero-subtitle">Upload any PDF and ask questions — powered by AI</p>
</div>
""", unsafe_allow_html=True)


# ---------------- Build RAG pipeline (cached per unique PDF) ----------------
@st.cache_resource(show_spinner="🔧 Building knowledge index...")
def build_rag_chain(pdf_bytes: bytes, pdf_name: str, api_key: str):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(pdf_bytes)
        tmp_path = tmp.name

    documents = PyPDFLoader(tmp_path).load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vectorstore = FAISS.from_documents(chunks, embeddings)

    llm = ChatGroq(model="openai/gpt-oss-20b", temperature=0, api_key=api_key)

    prompt_template = ChatPromptTemplate.from_template("""
You are a helpful assistant that answers questions based on the provided context.
Answer the question using ONLY the information in the CONTEXT below.
If the answer is not in the context, respond with: "I don't know based on the provided context."

CONTEXT:
{context}

QUESTION: {question}

ANSWER:
""")
    return vectorstore, llm, prompt_template, len(chunks), len(documents)


# ---------------- File uploader ----------------
uploaded_file = st.file_uploader(
    "📁 Drop your PDF here",
    type="pdf",
    help="Any PDF works — resume, article, research paper, contract, etc.",
)

if uploaded_file is None:
    st.markdown("""
    <div style="text-align:center; padding: 2rem; color: #718096;">
        <div style="font-size: 3rem;">⬆️</div>
        <p style="font-size: 1.1rem; font-weight: 500;">Upload a PDF above to get started</p>
        <p style="font-size: 0.85rem;">Supports any text-based PDF document</p>
    </div>
    """, unsafe_allow_html=True)

    # Builder badge shown on empty state
    st.markdown("""
    <div class="builder-badge">
        <p>✨ Built with passion by</p>
        <p class="builder-name">Totan</p>
        <p style="font-size:0.75rem; color:#888; margin-top:4px;">LangChain · Groq · FAISS · Streamlit</p>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# Build the pipeline
vectorstore, llm, prompt_template, chunk_count, page_count = build_rag_chain(
    uploaded_file.getvalue(),
    uploaded_file.name,
    GROQ_API_KEY,
)

# Success message with animation
st.markdown(f"""
<div style="background: linear-gradient(135deg, #ebf8ff, #bee3f8);
            border-left: 4px solid #3182ce;
            border-radius: 10px;
            padding: 0.8rem 1.2rem;
            margin-bottom: 1rem;
            animation: fadeInUp 0.4s ease;">
    <strong>✅ Ready!</strong> &nbsp;
    <span class="chip">📄 {page_count} pages</span>
    <span class="chip">✂️ {chunk_count} chunks</span>
    <span class="chip">🔢 384-dim embeddings</span>
</div>
""", unsafe_allow_html=True)


# ---------------- Sidebar ----------------
with st.sidebar:
    st.markdown("## ⚙️ Pipeline Info")
    st.markdown(f"""
    <div class="sidebar-card">
        <div style="font-size:0.75rem; color:#718096; text-transform:uppercase; letter-spacing:1px;">Document</div>
        <div style="font-weight:600; color:#2d3748; margin-top:2px; word-break:break-all;">{uploaded_file.name}</div>
    </div>
    <div class="sidebar-card">
        <div style="font-size:0.75rem; color:#718096; text-transform:uppercase; letter-spacing:1px;">Size</div>
        <div style="font-weight:600; color:#2d3748; margin-top:2px;">{uploaded_file.size / 1024:.1f} KB</div>
    </div>
    <div class="sidebar-card">
        <div style="font-size:0.75rem; color:#718096; text-transform:uppercase; letter-spacing:1px;">Pages · Chunks</div>
        <div style="font-weight:600; color:#2d3748; margin-top:2px;">{page_count} pages · {chunk_count} chunks</div>
    </div>
    <div class="sidebar-card">
        <div style="font-size:0.75rem; color:#718096; text-transform:uppercase; letter-spacing:1px;">Embeddings</div>
        <div style="font-weight:600; color:#2d3748; margin-top:2px;">all-MiniLM-L6-v2</div>
    </div>
    <div class="sidebar-card">
        <div style="font-size:0.75rem; color:#718096; text-transform:uppercase; letter-spacing:1px;">Vector Store</div>
        <div style="font-weight:600; color:#2d3748; margin-top:2px;">FAISS</div>
    </div>
    <div class="sidebar-card">
        <div style="font-size:0.75rem; color:#718096; text-transform:uppercase; letter-spacing:1px;">LLM</div>
        <div style="font-weight:600; color:#2d3748; margin-top:2px;">GPT-OSS 20B (Groq)</div>
    </div>

    <hr style="border:none; border-top:1px solid #e2e8f0; margin: 1.2rem 0;">

    <div style="text-align:center; padding: 0.5rem;">
        <div style="font-size:0.75rem; color:#718096;">Built by</div>
        <div style="font-size:1.1rem; font-weight:700;
                    background: linear-gradient(135deg, #667eea, #764ba2);
                    -webkit-background-clip: text;
                    -webkit-text-fill-color: transparent;
                    background-clip: text;">Totan</div>
    </div>
    """, unsafe_allow_html=True)


# ---------------- Q&A UI ----------------
question = st.text_input(
    "❓ Ask a question about your PDF:",
    placeholder="e.g. What is this document about?",
)

if question:
    try:
        with st.spinner("🤔 Thinking..."):
            retrieved_docs = vectorstore.similarity_search(question, k=3)
            context = "\n\n".join(doc.page_content for doc in retrieved_docs)
            filled_prompt = prompt_template.format(context=context, question=question)
            response = llm.invoke(filled_prompt)

        st.markdown(f"""
        <div class="answer-card">
            <div class="answer-label">💬 Answer</div>
            <div class="answer-text">{response.content}</div>
        </div>
        """, unsafe_allow_html=True)

        with st.expander("📎 View source chunks used"):
            for i, doc in enumerate(retrieved_docs, start=1):
                page = doc.metadata.get("page", "?")
                st.markdown(f"**Match {i}** — Page {page + 1 if isinstance(page, int) else page}")
                st.code(doc.page_content[:400], language=None)

    except Exception as e:
        st.error(f"❌ Error getting answer: {e}")

# ---------------- Footer ----------------
st.markdown("""
<div class="builder-badge">
    <p>✨ Crafted with ❤️ by</p>
    <p class="builder-name">Totan</p>
    <p style="font-size:0.75rem; color:#888; margin-top:4px;">
        LangChain &nbsp;·&nbsp; Groq &nbsp;·&nbsp; FAISS &nbsp;·&nbsp; HuggingFace &nbsp;·&nbsp; Streamlit
    </p>
</div>
""", unsafe_allow_html=True)
