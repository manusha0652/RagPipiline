# 🦀 Ferris the Librarian - Local Rust RAG Assistant

**Ferris the Librarian** is a 100% offline, free, and privacy-first Retrieval-Augmented Generation (RAG) chatbot built to answer questions about the Rust programming language. Ferris uses "The Rust Programming Language" book as his brain, strictly citing page numbers and refusing to hallucinate answers outside of his library context.

![Project Status](https://img.shields.io/badge/Status-Active-success)
![Tech Stack](https://img.shields.io/badge/Tech-Python_|_LangChain_|_Ollama_|_Streamlit-blue)

## ✨ Features
* **100% Local & Private**: No API keys required. No data is sent to OpenAI or anywhere else. Everything runs on your hardware.
* **Anti-Hallucination Constraints**: Ferris is strictly prompted to only answer based on the provided PDF context. If he doesn't know, he'll admit it!
* **Source Citations**: Every answer includes the exact page number from the source material.
* **Friendly Persona**: A welcoming, Rust-crustacean themed UI built cleanly with Streamlit.

## 🏗️ Architecture & Tech Stack
* **Language**: Python 3
* **Interface**: Streamlit
* **RAG Orchestration**: LangChain
* **Vector Database**: ChromaDB (Embedded locally)
* **Embeddings**: `nomic-embed-text` (Powered by Ollama)
* **LLM**: `llama3` (Powered by Ollama)

## 🚀 Getting Started

### 1. Prerequisites
* Python 3.8+
* [Ollama](https://ollama.com/) installed on your machine.

### 2. Setup your Local Models
Pull the required LLM and Embedding models into your local Ollama environment:
```bash
ollama pull llama3
ollama pull nomic-embed-text

3. Install Dependencies
Clone this repository, place The Rust Programming Language.pdf in the root folder, and set up your Python environment:

python -m venv venv
# Activate on Windows:
.\venv\Scripts\activate
# Activate on Mac/Linux:
source venv/bin/activate

pip install -r requirements.txt


4. Build the Knowledge Base (Phase 1)
Run the ingestion script to parse the PDF, split the text into smart chunks, calculate vector embeddings, and save them to a local SQLite/Chroma database.

python ingest.py



5. Chat with Ferris! (Phase 2)
Launch the Streamlit web interface:

streamlit run app.py
