---
title: PDF QA Bot
emoji: 📄
colorFrom: blue
colorTo: purple
sdk: streamlit
sdk_version: 1.50.0
app_file: streamlit_app.py
pinned: false
license: mit
---

# 📄 PDF QA Bot

A Retrieval-Augmented Generation (RAG) chatbot that answers questions about a PDF.

## Built With

- **LangChain** — RAG framework
- **Groq** — Fast LLM inference (`openai/gpt-oss-20b`)
- **HuggingFace Embeddings** — `sentence-transformers/all-MiniLM-L6-v2`
- **FAISS** — Vector similarity search
- **Streamlit** — Web UI

## How It Works

1. PDF is loaded and split into chunks
2. Each chunk is embedded into a 384-dim vector
3. Vectors are stored in a FAISS index
4. User question is embedded, top-K similar chunks retrieved
5. Groq LLM answers using the retrieved chunks as context

## Setup

Set the `GROQ_API_KEY` secret in Space settings.
