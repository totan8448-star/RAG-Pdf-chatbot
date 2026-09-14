import os
import streamlit as st
from dotenv import load_dotenv

# Import our custom modules
from rag_pipeline import build_rag_chain, generate_answer
import streamlit_ui as ui

# ---------------- Initialization ----------------
st.set_page_config(page_title="PDF QA Bot", page_icon="📄", layout="centered")
ui.load_custom_css()
load_dotenv()

def get_groq_api_key() -> str:
    # Check .env file / system environment first
    key = os.getenv("GROQ_API_KEY", "")
    
    # If not found, fall back safely to Streamlit secrets
    if not key:
        try:
            key = st.secrets.get("GROQ_API_KEY", "")
        except Exception:
            pass

    if not key:
        st.error("❌ GROQ_API_KEY not found. Add it to Streamlit secrets or a local .env file.")
        st.stop()
        
    return key

GROQ_API_KEY = get_groq_api_key()

@st.cache_resource(show_spinner="⚙️ Processing your file...")
def initialize_pipeline(file_bytes: bytes, file_name: str, api_key: str):
    return build_rag_chain(file_bytes, file_name, api_key)

# ---------------- Application Layout ----------------
ui.render_hero()

uploaded_file = st.file_uploader(
    "📁 Upload your file", 
    type=["pdf", "png", "jpg", "jpeg", "doc", "docx"]
)

if uploaded_file is None:
    st.markdown(
        '<div class="empty-chat">⬆️<br>Upload a PDF, Image, or Word doc to start chatting</div>', 
        unsafe_allow_html=True
    )
    ui.render_footer()
    st.stop()

# ---------------- RAG Setup ----------------
vectorstore, llm, prompt_template, chunk_count, page_count = initialize_pipeline(
    uploaded_file.getvalue(), 
    uploaded_file.name, 
    GROQ_API_KEY
)

ui.render_status_bar(uploaded_file.name, page_count, uploaded_file.size / 1024)
ui.render_sidebar(uploaded_file.name, page_count, uploaded_file.size / 1024)

# ---------------- Chat System ----------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

st.markdown(
    '<div class="chat-container"><div class="chat-header">🤖 Chat with your File</div></div>', 
    unsafe_allow_html=True
)

# Render history
if not st.session_state.chat_history:
    st.markdown('<div class="empty-chat">Ask anything about your file!</div>', unsafe_allow_html=True)
else:
    for msg in st.session_state.chat_history:
        ui.render_chat_message(msg["role"], msg["content"])

# User Input
with st.form(key="chat_form", clear_on_submit=True):
    col1, col2 = st.columns([5, 1])
    with col1:
        question = st.text_input(
            "question", 
            placeholder="Type your question here...", 
            label_visibility="collapsed"
        )
    with col2:
        send = st.form_submit_button("Send ➤", use_container_width=True, type="primary")

# Handle Submission
if send and question.strip():
    st.session_state.chat_history.append({"role": "user", "content": question.strip()})
    
    with st.spinner("🤔 Thinking..."):
        answer = generate_answer(vectorstore, llm, prompt_template, question.strip())

    st.session_state.chat_history.append({"role": "assistant", "content": answer})
    st.rerun()

if st.session_state.chat_history:
    if st.button("🗑️ Clear chat"):
        st.session_state.chat_history = []
        st.rerun()

ui.render_footer()