# Local RAG Pipeline — Ollama + ChromaDB

A fully local RAG (Retrieval-Augmented Generation) pipeline.
Ask questions to your PDF using local LLMs via Ollama. No cloud. No API costs.

## Tech Stack
- **Ollama** — local LLM + embeddings
- **ChromaDB** — persistent vector database
- **pypdf** — PDF text extraction
- **LangChain** — text chunking

## Setup

### 1. Install dependencies
pip install -r requirements.txt

### 2. Pull Ollama models
ollama pull nomic-embed-text
ollama pull llama3

### 3. Run
- Set your PDF path in `rag_chromadb.py` (PDF_PATH variable)
- Then run:
python rag_chromadb.py

## How it works
1. Loads and extracts text from a PDF
2. Splits text into chunks
3. Generates embeddings via Ollama
4. Stores vectors in ChromaDB (persisted to disk)
5. On question → retrieves relevant chunks → sends to LLM → returns answer

Step 4 — Push to GitHub:
bash# Inside your project folder
git init
git add .
git commit -m "Initial commit — Local RAG pipeline with Ollama + ChromaDB"

# Create a new repo on github.com, then:
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
git branch -M main
git push -u origin main
