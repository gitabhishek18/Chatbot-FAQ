# Chatbot-FAQ

A lightweight Streamlit-based Q&A project with two chatbot apps:

- **PDF Chatbot** — upload a PDF and ask questions about its contents.
- **YouTube Transcript Q&A Bot** — provide a YouTube link and chat with the video transcript.

The repository currently contains two Python apps: `chat2.py` and `youtube_qna_rag.py`. :contentReference[oaicite:0]{index=0}

---

## Features

### 1) PDF Chatbot
The PDF chatbot lets you upload a PDF file, splits the extracted text into chunks, creates embeddings, stores them in a FAISS vector index, and answers questions using an Ollama LLM. It uses Streamlit for the UI, PyPDF2 for PDF parsing, `RecursiveCharacterTextSplitter` for chunking, HuggingFace embeddings (`all-MiniLM-L6-v2`), FAISS for retrieval, and Ollama with the `llama3` model for answering questions. :contentReference[oaicite:1]{index=1}

### 2) YouTube Q&A Bot
The YouTube chatbot accepts a YouTube URL, extracts the video ID, fetches the transcript with `youtube_transcript_api`, chunks the transcript, embeds it using Ollama embeddings (`mxbai-embed-large`), stores it in FAISS, and answers user questions in a chat-style Streamlit interface using the `llama3` model. It also keeps chat history in Streamlit session state. :contentReference[oaicite:2]{index=2}

---

## Project Structure

```bash
Chatbot-FAQ/
├── chat2.py              # PDF question-answering chatbot
└── youtube_qna_rag.py   # YouTube transcript Q&A chatbot
