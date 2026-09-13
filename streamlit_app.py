"""
PDF QA Bot — Streamlit Web UI.

Same RAG pipeline as app.py, but with a browser interface.
Run with:  streamlit run streamlit_app.py
"""

import tempfile

import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

# ---------------- Page config ----------------
st.set_page_config(page_title="PDF QA Bot", page_icon="📄", layout="centered")
st.title("📄 PDF QA Bot")
st.caption("Ask questions about the PDF — powered by LangChain + Groq + FAISS")


# ---------------- Build RAG pipeline (cached per unique PDF) ----------------
@st.cache_resource(show_spinner="🔧 Building index for this PDF...")
def build_rag_chain(pdf_bytes: bytes, pdf_name: str):
    # Streamlit hashes pdf_bytes → same file re-uploaded hits cache instantly.
    # PyPDFLoader needs a file path, so we write the bytes to a temp file.
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(pdf_bytes)
        tmp_path = tmp.name

    # 1. Load & split
    documents = PyPDFLoader(tmp_path).load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = splitter.split_documents(documents)

    # 2. Embed & store
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vectorstore = FAISS.from_documents(chunks, embeddings)

    # 3. LLM + prompt
    llm = ChatGroq(model="openai/gpt-oss-20b", temperature=0)
    prompt_template = ChatPromptTemplate.from_template("""
You are a helpful assistant that answers questions based on the provided context.
Answer the question using ONLY the information in the CONTEXT below.
If the answer is not in the context, respond with: "I don't know based on the provided context."

CONTEXT:
{context}

QUESTION: {question}

ANSWER:
""")

    return vectorstore, llm, prompt_template, len(chunks)


# ---------------- File uploader ----------------
uploaded_file = st.file_uploader(
    "📁 Upload a PDF to ask questions about",
    type="pdf",
    help="Any PDF works — a resume, article, research paper, etc.",
)

if uploaded_file is None:
    st.info("👆 Upload a PDF above to get started.")
    st.stop()   # halt here until a file is uploaded

# Build (or fetch from cache) the pipeline for this specific PDF
vectorstore, llm, prompt_template, chunk_count = build_rag_chain(
    uploaded_file.getvalue(),
    uploaded_file.name,
)


# ---------------- Sidebar: pipeline info ----------------
with st.sidebar:
    st.header("⚙️ Pipeline")
    st.write(f"📄 **PDF:** `{uploaded_file.name}`")
    st.write(f"📏 **Size:** {uploaded_file.size / 1024:.1f} KB")
    st.write(f"✂️ **Chunks:** {chunk_count}")
    st.write("🔢 **Embeddings:** `all-MiniLM-L6-v2` (384-dim)")
    st.write("🗄️ **Vector store:** `FAISS`")
    st.write("🤖 **LLM:** `openai/gpt-oss-20b` (Groq)")


# ---------------- Q&A UI ----------------
question = st.text_input(
    "❓ Ask a question about the PDF:",
    placeholder="e.g. What is this document about?",
)

if question:
    try:
        with st.spinner("🤔 Thinking..."):
            # Retrieval
            retrieved_docs = vectorstore.similarity_search(question, k=3)
            context = "\n\n".join(doc.page_content for doc in retrieved_docs)

            # Generation
            filled_prompt = prompt_template.format(context=context, question=question)
            response = llm.invoke(filled_prompt)

        st.markdown("### 💬 Answer")
        st.success(response.content)

        with st.expander("📎 Sources used for this answer"):
            for i, doc in enumerate(retrieved_docs, start=1):
                page = doc.metadata.get("page", "?")
                st.markdown(f"**Match {i}** — Page {page}")
                st.text(doc.page_content[:400])
                st.divider()
    except Exception as e:
        st.error(f"❌ Error getting answer: {e}")
