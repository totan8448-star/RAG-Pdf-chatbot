"""
PDF QA Bot — built with LangChain + Groq + HuggingFace + FAISS.

Pipeline: Load PDF → Split into chunks → Embed → Store in FAISS
          → Retrieve relevant chunks for a question → Ask Groq LLM.
"""

import numpy as np
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

# ============================================================
# STEP 1: Load the PDF
# ============================================================
loader = PyPDFLoader("data/131310_Totan.pdf")
documents = loader.load()

print("\n--- STEP 1: PDF LOADED ---")
print(f"Number of pages: {len(documents)}")
print(f"First page metadata: {documents[0].metadata}")


# ============================================================
# STEP 2: Split into chunks
# ============================================================
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100,
)
chunks = splitter.split_documents(documents)

print("\n--- STEP 2: CHUNKS CREATED ---")
print(f"Original pages: {len(documents)}")
print(f"After splitting: {len(chunks)} chunks")
print(f"First chunk length: {len(chunks[0].page_content)} chars")


# ============================================================
# STEP 3: Create the embeddings model
# ============================================================
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

print("\n--- STEP 3: EMBEDDINGS MODEL READY ---")
print("Model: all-MiniLM-L6-v2 (384-dim vectors)")


# ============================================================
# STEP 4: Quick similarity demo (learning check)
# ============================================================
def cosine_similarity(v1, v2):
    v1, v2 = np.array(v1), np.array(v2)
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

demo_texts = [
    "I love programming in Python",
    "Python is my favorite coding language",
    "The pizza was delicious",
]
demo_vectors = embeddings.embed_documents(demo_texts)

print("\n--- STEP 4: SIMILARITY DEMO ---")
print(f"'Python programming' vs 'Python favorite lang': {cosine_similarity(demo_vectors[0], demo_vectors[1]):.4f}")
print(f"'Python programming' vs 'Pizza delicious':      {cosine_similarity(demo_vectors[0], demo_vectors[2]):.4f}")


# ============================================================
# STEP 5: Build the FAISS vector store from the chunks
# ============================================================
vectorstore = FAISS.from_documents(chunks, embeddings)

print("\n--- STEP 5: VECTOR STORE READY ---")
print(f"Total vectors stored: {vectorstore.index.ntotal}")


# ============================================================
# STEP 6: Initialize the Groq LLM
# ============================================================
llm = ChatGroq(model="openai/gpt-oss-20b", temperature=0)

print("\n--- STEP 6: GROQ LLM READY ---")
greeting = llm.invoke("Say hello in exactly 5 words.")
print(f"Groq says: {greeting.content}")


# ============================================================
# STEP 7: Build the prompt template (the "rules" for the LLM)
# ============================================================
prompt_template = ChatPromptTemplate.from_template("""
You are a helpful assistant that answers questions based on the provided context.
Answer the question using ONLY the information in the CONTEXT below.
If the answer is not in the context, respond with: "I don't know based on the provided context."

CONTEXT:
{context}

QUESTION: {question}

ANSWER:
""")

print("\n--- STEP 7: PROMPT TEMPLATE READY ---")


# ============================================================
# STEP 8: The RAG function — retrieve + generate
# ============================================================
def ask_pdf(question: str):
    # 1) Retrieve the top 3 most similar chunks from FAISS
    retrieved_docs = vectorstore.similarity_search(question, k=3)

    # 2) Join their text into one context string
    context = "\n\n".join(doc.page_content for doc in retrieved_docs)

    # 3) Fill the prompt template with context + question
    filled_prompt = prompt_template.format(context=context, question=question)

    # 4) Ask Groq and return the answer + source docs
    response = llm.invoke(filled_prompt)
    return response.content, retrieved_docs


# ============================================================
# STEP 9: Test the bot with real questions
# ============================================================
print(f"\n{'=' * 60}")
print("🤖  PDF QA BOT — TEST RUN")
print(f"{'=' * 60}")

test_questions = [
    "Who is this letter about?",
    "What is the GPN number mentioned?",
    "When was the letter issued?",
    "What is the weather in Tokyo today?",  # not in PDF → should say "I don't know"
]

for q in test_questions:
    answer, sources = ask_pdf(q)
    pages = [s.metadata.get("page") for s in sources]
    print(f"\n❓ Q: {q}")
    print(f"💬 A: {answer}")
    print(f"📎 Sources (pages): {pages}")
    print("-" * 60)
