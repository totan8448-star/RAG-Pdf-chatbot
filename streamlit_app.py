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
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Animated hero banner */
.hero-banner {
    background: linear-gradient(135deg, #4f46e5, #7c3aed, #db2777);
    background-size: 200% 200%;
    animation: gradientShift 6s ease infinite;
    border-radius: 20px;
    padding: 2.2rem 1.5rem 1.8rem;
    text-align: center;
    margin-bottom: 1.8rem;
    box-shadow: 0 10px 40px rgba(79, 70, 229, 0.35);
}

@keyframes gradientShift {
    0%   { background-position: 0% 50%; }
    50%  { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}

.hero-icon {
    font-size: 2.8rem;
    animation: bounce 2.5s ease-in-out infinite;
    display: inline-block;
}

@keyframes bounce {
    0%, 100% { transform: translateY(0); }
    50%       { transform: translateY(-8px); }
}

.hero-title {
    font-size: 2.2rem;
    font-weight: 700;
    color: white;
    margin: 0.3rem 0 0.2rem;
    letter-spacing: -0.5px;
}

.hero-sub {
    font-size: 0.95rem;
    color: rgba(255,255,255,0.8);
    font-weight: 300;
    margin: 0;
}

/* Upload box */
.upload-label {
    font-size: 0.9rem;
    font-weight: 600;
    color: #374151;
    margin-bottom: 0.3rem;
    display: block;
}

/* Success bar */
.status-bar {
    background: linear-gradient(135deg, #ecfdf5, #d1fae5);
    border: 1px solid #6ee7b7;
    border-radius: 12px;
    padding: 0.75rem 1.2rem;
    display: flex;
    align-items: center;
    gap: 0.6rem;
    margin-bottom: 1.5rem;
    animation: fadeSlide 0.4s ease;
}

.status-bar span {
    font-size: 0.85rem;
    font-weight: 600;
    color: #065f46;
}

.status-dot {
    width: 10px; height: 10px;
    background: #10b981;
    border-radius: 50%;
    animation: ping 1.5s ease-in-out infinite;
    flex-shrink: 0;
}

@keyframes ping {
    0%, 100% { opacity: 1; transform: scale(1); }
    50%       { opacity: 0.5; transform: scale(1.4); }
}

/* Chat area */
.chat-container {
    border: 1.5px solid #e5e7eb;
    border-radius: 16px;
    overflow: hidden;
    background: #fafafa;
    margin-bottom: 1rem;
    box-shadow: 0 4px 20px rgba(0,0,0,0.06);
}

.chat-header {
    background: linear-gradient(135deg, #4f46e5, #7c3aed);
    padding: 0.85rem 1.2rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

.chat-header-title {
    color: white;
    font-weight: 600;
    font-size: 0.95rem;
}

.chat-body {
    padding: 1.2rem;
    min-height: 80px;
}

/* Message bubbles */
.msg-user {
    display: flex;
    justify-content: flex-end;
    margin-bottom: 0.8rem;
    animation: fadeSlide 0.3s ease;
}

.msg-user-bubble {
    background: linear-gradient(135deg, #4f46e5, #7c3aed);
    color: white;
    border-radius: 18px 18px 4px 18px;
    padding: 0.7rem 1.1rem;
    max-width: 80%;
    font-size: 0.95rem;
    line-height: 1.5;
    box-shadow: 0 2px 10px rgba(79,70,229,0.3);
}

.msg-bot {
    display: flex;
    justify-content: flex-start;
    margin-bottom: 0.8rem;
    animation: fadeSlide 0.3s ease;
}

.msg-bot-avatar {
    width: 32px; height: 32px;
    background: linear-gradient(135deg, #4f46e5, #db2777);
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1rem;
    flex-shrink: 0;
    margin-right: 0.5rem;
    margin-top: 2px;
}

.msg-bot-bubble {
    background: white;
    color: #1f2937;
    border-radius: 18px 18px 18px 4px;
    padding: 0.7rem 1.1rem;
    max-width: 80%;
    font-size: 0.95rem;
    line-height: 1.6;
    box-shadow: 0 2px 10px rgba(0,0,0,0.08);
    border: 1px solid #f3f4f6;
}

@keyframes fadeSlide {
    from { opacity: 0; transform: translateY(10px); }
    to   { opacity: 1; transform: translateY(0); }
}

/* Empty chat state */
.empty-chat {
    text-align: center;
    padding: 1.5rem 1rem;
    color: #9ca3af;
}

.empty-chat-icon {
    font-size: 2.5rem;
    margin-bottom: 0.5rem;
}

/* Footer */
.footer {
    text-align: center;
    margin-top: 2.5rem;
    padding: 1.2rem;
    background: linear-gradient(135deg, #f5f3ff, #fdf2f8);
    border-radius: 14px;
    border: 1px solid #e9d5ff;
}

.footer-name {
    font-size: 1.05rem;
    font-weight: 700;
    background: linear-gradient(135deg, #4f46e5, #db2777);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

.footer-stack {
    font-size: 0.75rem;
    color: #9ca3af;
    margin-top: 3px;
}

/* Sidebar */
.sb-item {
    background: white;
    border-radius: 10px;
    padding: 0.65rem 0.9rem;
    margin-bottom: 0.5rem;
    border: 1px solid #f3f4f6;
    box-shadow: 0 1px 4px rgba(0,0,0,0.05);
}

.sb-label {
    font-size: 0.7rem;
    color: #9ca3af;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    font-weight: 600;
}

.sb-value {
    font-size: 0.9rem;
    font-weight: 600;
    color: #1f2937;
    margin-top: 1px;
    word-break: break-word;
}

#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# ---------------- Hero ----------------
st.markdown("""
<div class="hero-banner">
    <div class="hero-icon">📄</div>
    <h1 class="hero-title">PDF QA Bot</h1>
    <p class="hero-sub">Upload a PDF · Ask anything · Get instant answers</p>
</div>
""", unsafe_allow_html=True)


# ---------------- RAG pipeline ----------------
@st.cache_resource(show_spinner="⚙️ Processing your PDF...")
def build_rag_chain(pdf_bytes: bytes, pdf_name: str, api_key: str):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(pdf_bytes)
        tmp_path = tmp.name

    documents = PyPDFLoader(tmp_path).load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
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


# ---------------- Upload ----------------
uploaded_file = st.file_uploader(
    "📁 Upload your PDF",
    type="pdf",
    help="Any text-based PDF — resume, article, research paper, contract, etc.",
)

if uploaded_file is None:
    st.markdown("""
    <div style="text-align:center; padding:2rem 1rem; color:#9ca3af;">
        <div style="font-size:3rem; margin-bottom:0.5rem;">⬆️</div>
        <div style="font-size:1rem; font-weight:500; color:#6b7280;">Upload a PDF to start chatting</div>
        <div style="font-size:0.82rem; margin-top:0.3rem;">Supports any text-based PDF document</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("""
    <div class="footer">
        <div style="font-size:0.8rem; color:#6b7280;">✨ Built by</div>
        <div class="footer-name">Totan</div>
        <div class="footer-stack">LangChain · Groq · FAISS · HuggingFace · Streamlit</div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# Build pipeline
vectorstore, llm, prompt_template, chunk_count, page_count = build_rag_chain(
    uploaded_file.getvalue(), uploaded_file.name, GROQ_API_KEY,
)

# Ready status bar
st.markdown(f"""
<div class="status-bar">
    <div class="status-dot"></div>
    <span>✅ "{uploaded_file.name}" ready &nbsp;·&nbsp; {page_count} pages &nbsp;·&nbsp; {uploaded_file.size / 1024:.1f} KB</span>
</div>
""", unsafe_allow_html=True)


# ---------------- Sidebar (pipeline info only) ----------------
with st.sidebar:
    st.markdown("### ⚙️ About this App")
    st.markdown(f"""
    <div class="sb-item">
        <div class="sb-label">Document</div>
        <div class="sb-value">{uploaded_file.name}</div>
    </div>
    <div class="sb-item">
        <div class="sb-label">Pages</div>
        <div class="sb-value">{page_count}</div>
    </div>
    <div class="sb-item">
        <div class="sb-label">File Size</div>
        <div class="sb-value">{uploaded_file.size / 1024:.1f} KB</div>
    </div>
    <div class="sb-item">
        <div class="sb-label">AI Model</div>
        <div class="sb-value">GPT-OSS 20B via Groq</div>
    </div>
    <div class="sb-item">
        <div class="sb-label">Search Engine</div>
        <div class="sb-value">FAISS Vector Store</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <hr style="border:none; border-top:1px solid #f3f4f6; margin:1rem 0;">
    <div style="text-align:center; padding:0.3rem;">
        <div style="font-size:0.72rem; color:#9ca3af;">Built by</div>
        <div class="footer-name" style="font-size:1.1rem;">Totan</div>
    </div>
    """, unsafe_allow_html=True)


# ---------------- Chat UI ----------------

# Init chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Chat container
st.markdown("""
<div class="chat-container">
    <div class="chat-header">
        <span style="font-size:1.1rem;">🤖</span>
        <span class="chat-header-title">Chat with your PDF</span>
    </div>
</div>
""", unsafe_allow_html=True)

# Display message history
if st.session_state.chat_history:
    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            st.markdown(f"""
            <div class="msg-user">
                <div class="msg-user-bubble">{msg["content"]}</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="msg-bot">
                <div class="msg-bot-avatar">🤖</div>
                <div class="msg-bot-bubble">{msg["content"]}</div>
            </div>
            """, unsafe_allow_html=True)
else:
    st.markdown("""
    <div class="empty-chat">
        <div class="empty-chat-icon">💬</div>
        <div style="font-size:0.9rem; font-weight:500;">Ask anything about your PDF</div>
        <div style="font-size:0.8rem; margin-top:0.2rem;">e.g. "What is this document about?"</div>
    </div>
    """, unsafe_allow_html=True)

# Input + Send button
col1, col2 = st.columns([5, 1])
with col1:
    question = st.text_input(
        "question",
        placeholder="Type your question here...",
        label_visibility="collapsed",
        key="question_input",
    )
with col2:
    send = st.button("Send ➤", use_container_width=True, type="primary")

# Handle send
if (send or question) and question.strip():
    # Add user message
    st.session_state.chat_history.append({"role": "user", "content": question.strip()})

    with st.spinner("🤔 Thinking..."):
        try:
            retrieved_docs = vectorstore.similarity_search(question.strip(), k=3)
            context = "\n\n".join(doc.page_content for doc in retrieved_docs)
            filled_prompt = prompt_template.format(context=context, question=question.strip())
            response = llm.invoke(filled_prompt)
            answer = response.content
        except Exception as e:
            answer = f"Sorry, I encountered an error: {e}"

    # Add bot response
    st.session_state.chat_history.append({"role": "assistant", "content": answer})
    st.rerun()

# Clear chat button
if st.session_state.chat_history:
    if st.button("🗑️ Clear chat", use_container_width=False):
        st.session_state.chat_history = []
        st.rerun()

# ---------------- Footer ----------------
st.markdown("""
<div class="footer" style="margin-top:2rem;">
    <div style="font-size:0.8rem; color:#6b7280;">✨ Crafted with ❤️ by</div>
    <div class="footer-name">Totan</div>
    <div class="footer-stack">LangChain &nbsp;·&nbsp; Groq &nbsp;·&nbsp; FAISS &nbsp;·&nbsp; HuggingFace &nbsp;·&nbsp; Streamlit</div>
</div>
""", unsafe_allow_html=True)
