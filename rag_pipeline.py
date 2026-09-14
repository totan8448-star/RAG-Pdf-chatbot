import os
import tempfile
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from PIL import Image
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def load_document(file_bytes: bytes, file_name: str):
    """Detects file type and returns a list of Documents."""
    ext = os.path.splitext(file_name)[1].lower()
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    if ext == '.pdf':
        return PyPDFLoader(tmp_path).load()
    
    elif ext in ['.docx', '.doc']:
        return Docx2txtLoader(tmp_path).load()
    
    elif ext in ['.png', '.jpg', '.jpeg']:
        img = Image.open(tmp_path)
        extracted_text = pytesseract.image_to_string(img)
        if not extracted_text.strip():
            extracted_text = "No readable text found in this image."
        return [Document(page_content=extracted_text, metadata={"source": file_name})]
    
    else:
        raise ValueError(f"Unsupported file type: {ext}")

def build_rag_chain(file_bytes: bytes, file_name: str, api_key: str):
    """Processes the file and builds the FAISS vector store and LLM."""
    documents = load_document(file_bytes, file_name)
    
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