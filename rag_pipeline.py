import tempfile
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

def build_rag_chain(pdf_bytes: bytes, api_key: str):
    """Processes the PDF and builds the FAISS vector store and LLM."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(pdf_bytes)
        tmp_path = tmp.name

    # Load and split
    documents = PyPDFLoader(tmp_path).load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = splitter.split_documents(documents)

    # Embeddings and Vectorstore
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    
    # LLM and Prompt
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

def generate_answer(vectorstore, llm, prompt_template, question: str) -> str:
    """Retrieves context and generates an answer."""
    try:
        retrieved_docs = vectorstore.similarity_search(question, k=3)
        context = "\n\n".join(doc.page_content for doc in retrieved_docs)
        filled_prompt = prompt_template.format(context=context, question=question)
        response = llm.invoke(filled_prompt)
        return response.content
    except Exception as e:
        return f"Sorry, I encountered an error: {e}"