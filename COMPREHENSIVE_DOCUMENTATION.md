# 📚 Comprehensive RAG Application Documentation

**Complete Guide: From Basics to Production**

---

## Table of Contents

1. [Core Concepts](#core-concepts)
2. [Project Architecture](#project-architecture)
3. [Folder Structure & File Guide](#folder-structure--file-guide)
4. [Prerequisites & Requirements](#prerequisites--requirements)
5. [Installation & Setup](#installation--setup)
6. [Running the Application](#running-the-application)
7. [System Components Deep Dive](#system-components-deep-dive)
8. [How It Works: The RAG Pipeline](#how-it-works-the-rag-pipeline)
9. [Configuration Guide](#configuration-guide)
10. [Feature Implementation Guide](#feature-implementation-guide)
11. [Troubleshooting Guide](#troubleshooting-guide)
12. [Advanced Topics](#advanced-topics)

---

# 1. Core Concepts

## 1.1 What is RAG (Retrieval-Augmented Generation)?

**RAG** is an AI technique that combines two powerful approaches:

### **Retrieval (Information Lookup)**
- Search through your documents to find relevant information
- Similar to a smart search engine that understands meaning, not just keywords
- Uses vector embeddings (mathematical representations of text)
- Finds the most similar documents to your question

### **Augmentation (Context Enhancement)**
- Takes the retrieved information and adds it to the question
- Provides the LLM with factual, domain-specific context
- Ensures the LLM generates answers based on your documents

### **Generation (Answer Creation)**
- Uses a Large Language Model (LLM) to generate the final answer
- The LLM reads both the question AND the retrieved context
- Produces a coherent answer grounded in your documents

**Benefits:**
- ✅ Accurate answers based on your specific documents
- ✅ No need for expensive model retraining
- ✅ Works with private/local data
- ✅ Combines local knowledge with LLM reasoning
- ✅ Reduces "hallucinations" (false information from LLM)

### **Real-World Example:**
```
User Question: "What is ISO 27001?"

Step 1: RETRIEVE
└─ Search vector database for relevant sections
└─ Find: "ISO 27001 is an information security management system standard..."

Step 2: AUGMENT  
└─ Combine with question:
   "Using this context: 'ISO 27001 is an information security management system...'
    Answer the question: 'What is ISO 27001?'"

Step 3: GENERATE
└─ LLM produces: "ISO 27001 is an international standard for information security
   management systems that helps organizations manage security of assets such as
   financial information, intellectual property, employee details, or information
   entrusted to you by clients."
```

---

## 1.2 Key Technologies Used

### **FAISS (Facebook AI Similarity Search)**
- Fast vector similarity search library
- Converts text to numerical vectors (embeddings)
- Uses distance metrics to find similar documents
- Scales to billions of vectors

### **Sentence Transformers (all-MiniLM-L6-v2)**
- Pre-trained model that converts text to embeddings
- Compact and efficient (384-dimensional embeddings)
- Understands semantic meaning (similar meaning = similar vectors)
- No GPU required

### **Ollama**
- Local LLM inference engine
- Runs language models on your machine
- Privacy-first (no data sent to cloud)
- Supports: Llama2, Mistral, Neural Chat, Orca Mini, etc.

### **FastAPI**
- Modern Python web framework
- Lightning-fast REST API creation
- Built-in data validation
- Automatic API documentation

### **Flask**
- Python web framework for the user interface
- Serves HTML/CSS/JavaScript
- Proxies requests to FastAPI backend

### **PyPDF**
- PDF document reading and text extraction
- Supports complex PDF structures
- Extracts text from multiple pages

---

# 2. Project Architecture

## 2.1 System Architecture Diagram

```
┌──────────────────────────────────────────────────────────────────┐
│                        USER BROWSER                              │
│                  http://localhost:5000                           │
│              (Beautiful Web Chat Interface)                      │
└──────────────────────────────┬─────────────────────────────────┘
                               │
                    HTTP Request (Question)
                               │
                               ▼
┌──────────────────────────────────────────────────────────────────┐
│                    FLASK WEB SERVER                              │
│                  Port 5000 (Frontend Layer)                      │
│                                                                  │
│  Responsibilities:                                              │
│  • Serve HTML/CSS/JavaScript UI                                │
│  • Handle browser requests                                      │
│  • Proxy API calls to FastAPI backend                          │
└──────────────────────────────┬─────────────────────────────────┘
                               │
                    HTTP POST /ask (Question)
                               │
                               ▼
┌──────────────────────────────────────────────────────────────────┐
│                   FASTAPI REST API                               │
│                  Port 8000 (Backend Layer)                       │
│                                                                  │
│  Endpoints:                                                      │
│  • GET  /health        → Check system status                    │
│  • POST /ask           → Process question (main endpoint)       │
└──────────────────────────────┬─────────────────────────────────┘
                               │
                ┌──────────────┴──────────────┐
                │                             │
                ▼                             ▼
    ┌──────────────────────┐      ┌────────────────────┐
    │ RAG PIPELINE MODULE  │      │  CONFIGURATION     │
    │                      │      │                    │
    │ ┌────────────────┐   │      │ • Model name       │
    │ │ PDF Reader     │   │      │ • Chunk size       │
    │ │ Text Processor │   │      │ • Ollama URL       │
    │ │ Vector DB      │   │      │ • Top-K results    │
    │ │ LLM Interface  │   │      └────────────────────┘
    │ └────────────────┘   │
    └──────────┬───────────┘
               │
    ┌──────────┴──────────┬──────────────┐
    │                     │              │
    ▼                     ▼              ▼
┌─────────┐      ┌──────────────┐  ┌──────────┐
│  FAISS  │      │  Embeddings  │  │  Ollama  │
│  Vector │      │  Model       │  │   LLM    │
│   DB    │      │              │  │          │
└─────────┘      └──────────────┘  └──────────┘
```

## 2.2 Data Flow

```
PDF Document (example/iso27001.pdf)
        │
        ▼
    [Load PDF]
        │
        ▼
    [Extract Text] ─────────── Raw text from all pages
        │
        ▼
    [Split into Chunks] ────── 150-word chunks with 30-word overlap
        │
        ▼
    [Convert to Embeddings] ── Sentence Transformers converts each chunk
        │                       to a 384-dimensional vector
        ▼
    [Store in FAISS] ─────────── Vector database for fast similarity search
        │
        ▼
 ┌──────────────────────────────────────────────────────────┐
 │              READY FOR QUERIES                           │
 └──────────────────────────────────────────────────────────┘
        │
        ▼
    User asks: "What is ISO 27001?"
        │
        ▼
    [Convert Question to Embedding]
        │
        ▼
    [Search FAISS] ─────────────── Find top-3 most similar chunks
        │                           using cosine similarity
        ▼
    [Retrieve Contexts] ─────────── Get actual text of matched chunks
        │
        ▼
    [Build Prompt] ────────────────
        │
        │  Combined prompt:
        │  "Answer using ONLY this context:
        │   [Retrieved Context 1]
        │   [Retrieved Context 2]
        │   [Retrieved Context 3]
        │   Question: What is ISO 27001?
        │   Answer:"
        │
        ▼
    [Send to Ollama LLM]
        │
        ▼
    [Generate Answer] ────────────── LLM generates coherent response
        │                             based on contexts
        ▼
    [Return to User] ────────────── Answer + contexts displayed in UI
```

---

# 3. Folder Structure & File Guide

## 3.1 Complete Project Directory Tree

```
RAG_FAISS_LLAMA/
│
├── 📄 Root Configuration Files (Application Settings)
│   ├── app.py                    # FastAPI backend (REST API, port 8000)
│   ├── app_flask.py              # Flask frontend (Web UI, port 5000)
│   ├── requirements.txt           # Python package dependencies
│   ├── pyproject.toml             # Project metadata & version info
│   ├── .env                       # Environment variables (model config)
│   └── .gitignore                 # Git ignore patterns
│
├── 📦 RAG Pipeline Module (Core Logic)
│   └── rag/
│       ├── __init__.py            # Package initialization
│       ├── config.py              # Configuration constants & settings
│       ├── data_loader.py         # PDF reading & text extraction
│       ├── text_utils.py          # Text chunking & preprocessing
│       ├── vector_store.py        # FAISS vector database setup
│       ├── llm.py                 # Ollama LLM interface
│       └── pipeline.py            # RAG orchestration (main logic)
│
├── 🎨 Frontend Files
│   └── templates/
│       └── index.html             # Web UI (HTML/CSS/JavaScript)
│
├── 📚 Example Data
│   └── example/
│       └── iso27001.pdf           # Sample PDF document for testing
│
├── 🐍 Python Virtual Environment
│   └── rfl/                       # venv with all dependencies installed
│       ├── Scripts/               # Activation scripts
│       ├── Lib/                   # Installed packages
│       └── pyvenv.cfg             # venv configuration
│
└── 📖 Documentation
    ├── README.md                  # Project overview & architecture
    ├── QUICKSTART.md              # Quick setup & run guide
    ├── SETUP_COMPLETE.md          # Detailed setup report
    └── COMPREHENSIVE_DOCUMENTATION.md (THIS FILE)
```

---

## 3.2 Detailed File Descriptions

### **app.py** - FastAPI Backend
```
Purpose: REST API server that handles all backend logic
Port: 127.0.0.1:8000
Key Features:
  - Lazy initialization: PDF loads on first request (not startup)
  - Thread-safe: Prevents concurrent initialization issues
  - Two endpoints: /health (status check) and /ask (main API)
  - Automatic API documentation at /docs

What it does:
  1. Receives POST request with question
  2. Initializes RAG pipeline (if first request)
  3. Retrieves relevant document chunks
  4. Sends to Ollama LLM
  5. Returns answer + contexts as JSON

Example Request:
  POST /ask
  {
    "question": "What is ISO 27001?"
  }

Example Response:
  {
    "answer": "ISO 27001 is an international standard...",
    "contexts": [
      "ISO 27001 is an information security management system...",
      "It provides a systematic approach to...",
      "Organizations implement ISO 27001 to..."
    ]
  }
```

### **app_flask.py** - Flask Frontend
```
Purpose: Web UI server and API proxy
Port: 127.0.0.1:5000
Key Features:
  - Serves beautiful chat interface (index.html)
  - Proxies requests to FastAPI backend
  - Handles errors gracefully
  - Real-time conversation UI

Routes:
  GET  /              → Serves index.html (chat interface)
  GET  /api/status    → Check backend health
  POST /api/ask       → Accepts user question, proxies to FastAPI
```

### **rag/config.py** - Configuration Settings
```
Purpose: Centralized configuration for the entire application
All settings can be overridden via environment variables

Key Settings:
  MODEL_NAME = "mistral"
    └─ Which Ollama model to use (options: llama2, mistral, neural-chat, orca-mini)

  OLLAMA_BASE_URL = "http://127.0.0.1:11434"
    └─ URL where Ollama service is running

  CHUNK_SIZE = 150
    └─ How many words to include in each chunk
    └─ Smaller = more specific contexts, but more retrieval
    └─ Larger = broader contexts, but less relevant

  CHUNK_OVERLAP = 30
    └─ Words to overlap between consecutive chunks
    └─ Prevents losing information at chunk boundaries
    └─ Overlap = CHUNK_SIZE - CHUNK_OVERLAP = step size

  TOP_K = 3
    └─ Number of document chunks to retrieve for each question
    └─ Larger = more context but slower processing
    └─ Smaller = faster but might miss relevant info

Environment Variables:
  MODEL_NAME=llama3 python app.py          # Use Llama3 model
  CHUNK_SIZE=200 TOP_K=5 python app.py     # Larger chunks, more contexts
```

### **rag/data_loader.py** - PDF Reader
```
Purpose: Extract text from PDF documents
Main Class: PDFReader

Code Flow:
  1. Initialize with PDF path: reader = PDFReader("path/to/file.pdf")
  2. Call read() method
  3. Iterates through all pages
  4. Extracts text from each page using pypdf library
  5. Returns list of all page texts

Example Usage:
  reader = PDFReader("example/iso27001.pdf")
  pages = reader.read()
  print(f"Extracted {len(pages)} pages")
```

### **rag/text_utils.py** - Text Chunking
```
Purpose: Split large documents into smaller chunks
Main Function: text2chunk(text, chunk_size, overlap)

Algorithm:
  1. Split text into words
  2. Create sliding window of chunk_size words
  3. Move window by (chunk_size - overlap) words each time
  4. Return list of text chunks

Example:
  Text: "The quick brown fox jumps over the lazy dog"
  Words: ["The", "quick", "brown", "fox", "jumps", "over", "the", "lazy", "dog"]
  chunk_size=3, overlap=1
  
  Chunks:
    1. "The quick brown"
    2. "brown fox jumps"
    3. "fox jumps over"
    4. "over the lazy"
    5. "the lazy dog"

Why Chunking?
  ✅ FAISS works better with reasonable-length texts
  ✅ Enables precise retrieval (one chunk per question answer)
  ✅ Saves memory (load only needed chunks, not entire PDF)
  ✅ Improves answer quality (focused context)
```

### **rag/vector_store.py** - FAISS Vector Database
```
Purpose: Create and manage vector search database
Main Class: FAISSRetrieval

How it works:
  1. Takes list of text chunks as input
  2. Uses Sentence Transformer to convert each chunk to embedding
  3. Stores embeddings in FAISS index
  4. When searching, converts query to embedding
  5. Uses FAISS to find most similar chunks (fast similarity search)

Embedding Model: all-MiniLM-L6-v2
  - Converts any text to 384-dimensional vector
  - Similar texts have similar vectors (high cosine similarity)
  - No GPU required, fast on CPU

Search Algorithm: IndexFlatL2 (Euclidean distance)
  - Searches all vectors for closest matches
  - L2 distance: √((x₁-x₂)² + (y₁-y₂)² + ... + (z₁-z₂)²)
  - Simple but accurate, good for datasets < 1M vectors

Example Usage:
  from rag.vector_store import FAISSRetrieval
  
  chunks = ["ISO 27001 is...", "It covers...", "Organizations implement..."]
  retrieval = FAISSRetrieval(chunks)
  
  results = retrieval.retrieve("What is ISO 27001?", top_k=3)
  # Returns: ["ISO 27001 is...", "It covers...", "Organizations implement..."]
```

### **rag/llm.py** - LLM Interface
```
Purpose: Communicate with Ollama LLM service
Main Class: OllamaLLM

Features:
  - Calls Ollama HTTP API to generate text
  - Supports custom timeout (default 30 seconds)
  - Includes error handling and logging
  - Returns generated text as string

Methods:
  __init__(model_name, base_url, timeout=30)
    └─ Initialize with model name and Ollama URL

  generate(prompt: str) → str
    └─ Send prompt to Ollama, get generated answer
    └─ Includes debugging logs for troubleshooting

Generation Parameters:
  model: Which Ollama model to use
  prompt: The prompt text to send
  stream: False (get complete response at once)
  num_predict: 200 (limit output to 200 tokens for faster generation)

Example Usage:
  llm = OllamaLLM("mistral", "http://127.0.0.1:11434")
  answer = llm.generate("What is AI?")
  print(answer)

Error Handling:
  - ConnectionError: Ollama service not running
  - TimeoutError: Ollama taking too long to respond
  - HTTP errors: Invalid request to Ollama
```

### **rag/pipeline.py** - RAG Pipeline
```
Purpose: Orchestrates the entire RAG workflow
Main Class: SimpleRAGPipeline

Architecture:
  ┌─────────────────────────────────────────────────┐
  │  SimpleRAGPipeline(retrieval, llm, top_k)       │
  │                                                 │
  │  Components:                                    │
  │  • retrieval: FAISSRetrieval instance           │
  │  • llm: OllamaLLM instance                      │
  │  • top_k: Number of contexts to retrieve       │
  └─────────────────────────────────────────────────┘

Main Method: run(query: str) → Answer
  Step 1: RETRIEVE
    ├─ Call retrieval.retrieve(query, top_k)
    └─ Get top-k most similar document chunks

  Step 2: AUGMENT
    ├─ Combine chunks with the question
    ├─ Create optimized prompt
    └─ Ensure LLM focuses on provided context

  Step 3: GENERATE
    ├─ Call llm.generate(prompt)
    ├─ LLM produces answer based on context
    └─ Handle errors and timeouts

  Step 4: RETURN
    ├─ Create Answer object with:
    │  ├─ answer: Generated text
    │  └─ contexts: Retrieved document chunks
    └─ Return to API

Answer Class:
  Contains:
    - answer: str (the generated answer)
    - contexts: list[str] (the retrieved contexts)

Example Usage:
  pipeline = SimpleRAGPipeline(retrieval, llm, top_k=3)
  result = pipeline.run("What is ISO 27001?")
  print(f"Answer: {result.answer}")
  print(f"Contexts: {result.contexts}")
```

### **templates/index.html** - Web UI
```
Purpose: Beautiful chat interface for users
Features:
  ✅ Modern, responsive design
  ✅ Chat-like conversation display
  ✅ Real-time message streaming (if implemented)
  ✅ Error message display
  ✅ Loading indicators
  ✅ Context display (source documents shown)
  ✅ Mobile-friendly layout

Technologies:
  - HTML5: Semantic markup
  - CSS3: Modern styling with gradients & animations
  - JavaScript: Real-time interaction & API calls
  - Fetch API: Async communication with backend

Main Functions:
  - sendMessage(): Send question to backend
  - displayMessage(): Show message in chat
  - displayContexts(): Show source documents
  - showError(): Display error messages
  - clearChat(): Reset conversation
```

---

# 4. Prerequisites & Requirements

## 4.1 System Requirements

| Requirement | Minimum | Recommended |
|-----------|---------|-------------|
| **RAM** | 4 GB | 8+ GB |
| **Disk Space** | 2 GB free | 5+ GB free |
| **CPU** | Dual-core | 4+ cores |
| **OS** | Windows/Mac/Linux | Any (tested on Windows 11) |
| **GPU** | Not required | Optional (speeds up Ollama) |

## 4.2 Software Requirements

### **Required**
1. **Python 3.11+**
   - Download: https://www.python.org/downloads/
   - Check: `python --version`
   - Windows: Add to PATH during installation

2. **Ollama**
   - Download: https://ollama.ai
   - Install: Follow platform-specific instructions
   - Running: `ollama serve` or Windows service
   - Model: `ollama pull mistral` (or llama2)

3. **Git** (optional, for version control)
   - Download: https://git-scm.com/

### **Optional but Useful**
- Visual Studio Code (editor)
- PyCharm Professional (IDE)
- Docker (containerization)

## 4.3 Python Dependencies

All automatically installed via `pip install -r requirements.txt`:

```
fastapi==0.95.0+           # REST API framework
uvicorn==0.21.0+           # ASGI server
flask==3.0.0+              # Web framework for UI
pypdf==3.0.0+              # PDF reading
faiss-cpu==1.17.0+         # Vector similarity search
sentence-transformers==2.2.0+ # Text embeddings
requests==2.31.0+          # HTTP requests
python-dotenv==1.0.0+      # Environment variables
numpy==1.24.0+             # Numerical computing
torch==2.0.0+              # Pytorch (dependency of sentence-transformers)
```

### **Dependency Details:**

| Package | Purpose | Size | Notes |
|---------|---------|------|-------|
| FastAPI | REST API framework | ~50 MB | Lightweight, fast |
| Flask | Web UI server | ~20 MB | Standard Python web framework |
| PyPDF | PDF reading | ~10 MB | Pure Python, no dependencies |
| FAISS | Vector search | ~500 MB | CPU version sufficient |
| Sentence Transformers | Text embeddings | ~500 MB | Downloads model on first use |
| PyTorch | ML library | ~1-2 GB | Required by sentence-transformers |
| NumPy | Numerical computing | ~100 MB | Used by FAISS |
| Requests | HTTP library | ~5 MB | Simple HTTP client |
| python-dotenv | Config management | ~5 MB | Loads .env files |

## 4.4 Network Requirements

- **Ollama Service**: `http://127.0.0.1:11434` (localhost, configurable)
- **FastAPI Backend**: `http://127.0.0.1:8000` (localhost)
- **Flask Frontend**: `http://127.0.0.1:5000` (localhost)
- **No internet required**: Everything runs locally

## 4.5 File Requirements

Before running:
1. **PDF Document**: `example/iso27001.pdf` (or any PDF)
2. **Configuration**: `rag/config.py` or `.env`
3. **Templates**: `templates/index.html`

---

# 5. Installation & Setup

## 5.1 Step-by-Step Installation

### **Step 1: Install Ollama**

**Windows:**
1. Visit https://ollama.ai
2. Download Ollama installer
3. Run installer, follow prompts
4. Check installation: Open PowerShell and type `ollama --version`

**macOS:**
```bash
brew install ollama
```

**Linux:**
```bash
curl https://ollama.ai/install.sh | sh
```

### **Step 2: Download LLM Model**

Open terminal/PowerShell and run:
```bash
ollama pull mistral
```

Or for other models:
```bash
ollama pull llama2           # Better quality but slower
ollama pull neural-chat      # Fast, good for chat
ollama pull orca-mini        # Very fast, smaller model
```

Check downloaded models:
```bash
ollama list
```

### **Step 3: Install Python**

1. Download Python 3.11+ from https://www.python.org/downloads/
2. Run installer
3. ✅ **Important**: Check "Add Python to PATH"
4. Verify installation:
   ```bash
   python --version
   # Output: Python 3.11.x (or higher)
   ```

### **Step 4: Clone/Download Project**

Option A: Download ZIP
- Visit repository
- Download ZIP
- Extract to `C:\Users\YOUR_USERNAME\Documents\GenAI\RAG_FAISS_LLAMA\`

Option B: Git Clone
```bash
cd C:\Users\YOUR_USERNAME\Documents
git clone <repository-url> RAG_FAISS_LLAMA
cd RAG_FAISS_LLAMA
```

### **Step 5: Create Virtual Environment**

Navigate to project folder:
```bash
cd C:\Users\YOUR_USERNAME\Documents\GenAI\RAG_FAISS_LLAMA
```

Create virtual environment (recommended):
```bash
python -m venv rfl
```

This creates the `rfl` folder with isolated Python environment.

### **Step 6: Activate Virtual Environment**

**Windows PowerShell:**
```powershell
.\rfl\Scripts\Activate.ps1
# If error: Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
```

**Windows Command Prompt:**
```cmd
rfl\Scripts\activate.bat
```

**macOS/Linux:**
```bash
source rfl/bin/activate
```

**Success indicator**: Your terminal prompt starts with `(rfl)`

### **Step 7: Install Dependencies**

With virtual environment activated:
```bash
pip install -r requirements.txt
```

This installs all Python packages. Takes 5-10 minutes depending on internet.

Progress check:
- Watch for "Successfully installed" messages
- Final line should say "Successfully installed [N] packages"


### **Step 8: Configure Application**

Edit `.env` file in project root:
```env
MODEL_NAME=mistral
OLLAMA_BASE_URL=http://127.0.0.1:11434
CHUNK_SIZE=150
CHUNK_OVERLAP=30
TOP_K=3
```

Or edit `rag/config.py` directly:
```python
MODEL_NAME = os.getenv("MODEL_NAME", "mistral")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 150))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 30))
TOP_K = int(os.getenv("TOP_K", 3))
```

### **Step 9: Verify Setup**

Run verification script:
```bash
python -c "
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import requests

print('✓ FAISS installed')
print('✓ NumPy installed')
model = SentenceTransformer('all-MiniLM-L6-v2')
print('✓ Sentence Transformers installed & model available')
print('Setup verification complete!')
"
```

---

## 5.2 Quick Setup Checklist

- [ ] Python 3.11+ installed (`python --version`)
- [ ] Ollama installed (`ollama --version`)
- [ ] Ollama model downloaded (`ollama list`)
- [ ] Project folder exists
- [ ] Virtual environment created (`rfl` folder exists)
- [ ] Virtual environment activated (prompt shows `(rfl)`)
- [ ] Requirements installed (`pip list | find fastapi`)
- [ ] `.env` or `rag/config.py` configured
- [ ] PDF in `example/` folder or path configured

---

# 6. Running the Application

## 6.1 Quick Start (Automatic - Recommended)

**Option 1: One Command**

```powershell
cd C:\Users\VISRUTHI\Documents\GenAI\RAG_FAISS_LLAMA
.\rfl\Scripts\Activate.ps1
python run.py
```

Wait 10-15 seconds for startup, then open: **http://localhost:5000**

**Option 2: Make it Easier**

Create batch file `start.bat` in project root:
```batch
@echo off
cd /d %~dp0
call rfl\Scripts\activate.bat
python run.py
```

Then just double-click `start.bat` to start everything!

---

## 6.2 Manual Startup (Two Terminals - More Control)

This is useful if you want to see debug output from each component.

### **Terminal 1: Start Ollama Service**

```powershell
# If Ollama installed as service (Windows):
# It may already be running. Check:
# Open http://127.0.0.1:11434/api/tags in browser

# Or manually start:
ollama serve
```

**Success indicators:**
- Output shows "serving on 127.0.0.1:11434"
- No errors

### **Terminal 2: Start FastAPI Backend**

```powershell
cd C:\Users\VISRUTHI\Documents\GenAI\RAG_FAISS_LLAMA
.\rfl\Scripts\Activate.ps1

python -m uvicorn app:app --host 127.0.0.1 --port 8000 --reload
```

**Success indicators:**
- Output shows "Uvicorn running on http://127.0.0.1:8000"
- Shows "Reload enabled, watching for file changes"

### **Terminal 3: Start Flask Frontend**

```powershell
cd C:\Users\VISRUTHI\Documents\GenAI\RAG_FAISS_LLAMA
.\rfl\Scripts\Activate.ps1

python app_flask.py
```

**Success indicators:**
- Output shows "Running on http://127.0.0.1:5000"
- Shows "WARNING in werkzeug" (normal)

### **Step 4: Open Browser**

Visit: **http://localhost:5000**

You should see:
1. Beautiful chat interface
2. Input box for questions
3. Chat messages area
4. Source contexts below answers

---

## 6.3 First Test Query

1. In the chat box, type:
   ```
   What is ISO 27001?
   ```

2. Click "Send" or press Enter

3. **First request takes 10-30 seconds** because:
   - PDF is being loaded
   - Text is being chunked
   - Embeddings are being generated
   - FAISS index is being created
   - LLM is being initialized

4. You should see:
   - Answer displayed in chat
   - Source documents shown below
   - Response time in chat bubble

---

## 6.4 Stopping the Application

**Method 1: Close terminals**
- Press `Ctrl+C` in each terminal (FastAPI, Flask, Ollama)
- Close browser tab

**Method 2: Task Manager**
```powershell
# Kill Python processes
taskkill /F /IM python.exe

# Kill Ollama
taskkill /F /IM ollama.exe
```

---

# 7. System Components Deep Dive

## 7.1 Component 1: Data Loader (PDF Reading)

### **File**: `rag/data_loader.py`

### **What it does:**
Reads PDF files and extracts all text content.

### **How it works:**

```
PDF File
  │
  ├─ Page 1
  │  └─ [Image, Text, Tables, Formatting]
  │
  ├─ Page 2
  │  └─ [Image, Text, Tables, Formatting]
  │
  └─ Page N
     └─ [Image, Text, Tables, Formatting]
  
       ↓ (PyPDF processes each page)
  
  ├─ Page 1 extracted text
  ├─ Page 2 extracted text
  └─ Page N extracted text
  
       ↓ (Join all text)
  
  Single concatenated string with all text
```

### **Implementation:**

```python
from pypdf import PdfReader

class PDFReader:
    def __init__(self, path):
        self.path = path  # Path to PDF file
    
    def read(self):
        # 1. Open PDF file
        reader = PdfReader(self.path)
        
        # 2. Loop through all pages
        text = []
        for page in reader.pages:
            # 3. Extract text from each page
            t = page.extract_text()
            if t:  # Only add if text extracted
                text.append(t)
        
        # 4. Join all page texts together
        return ["\n".join(text)]
```

### **Key Points:**
- Returns list with single element (entire PDF as one string)
- Skips empty pages (pages with no text)
- Joins pages with newline separator
- Supports complex PDF structures

### **Limitations:**
- Cannot extract from scanned PDFs (images)
- May struggle with unusual formatting
- Doesn't extract images or tables (text only)

### **Usage:**

```python
from rag.data_loader import PDFReader

reader = PDFReader("example/iso27001.pdf")
content = reader.read()  # content is a list
full_text = content[0]   # Get the single element

print(f"Extracted {len(full_text)} characters")
```

---

## 7.2 Component 2: Text Processing (Chunking)

### **File**: `rag/text_utils.py`

### **What it does:**
Splits large text into manageable chunks for embedding and retrieval.

### **Why chunking is needed:**

```
Problem: Entire PDF is 100,000+ words
  ├─ Too long to embed efficiently
  ├─ Loses context when searching
  └─ Cannot identify specific relevant sections

Solution: Break into 150-word chunks
  ├─ Each chunk = focused topic
  ├─ Fast embedding and searching
  └─ Can retrieve exact relevant section
```

### **Chunking Algorithm:**

```
Original Text:
"The quick brown fox jumps over the lazy dog is amazing..."

Words:
["The", "quick", "brown", "fox", "jumps", "over", "the", "lazy", "dog", "is", "amazing", ...]

Parameters:
  chunk_size = 3   (words per chunk)
  overlap = 1      (words to share between chunks)
  step = chunk_size - overlap = 2

Process:
Position 0: Take words [0:3]   → "The quick brown"
Position 2: Take words [2:5]   → "brown fox jumps"
Position 4: Take words [4:7]   → "jumps over the"
Position 6: Take words [6:9]   → "the lazy dog"
Position 8: Take words [8:11]  → "dog is amazing"

Result: 5 chunks with overlapping words
```

### **Implementation:**

```python
def text2chunk(text: str, chunk_size: int, overlap: int = 0):
    # 1. Split text into words
    words = text.split()
    
    # 2. Calculate step size
    step = chunk_size - overlap
    
    # 3. Create chunks
    chunks = []
    for i in range(0, len(words), step):
        # Take chunk_size words starting at position i
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
    
    return chunks
```

### **Configuration:**

| Setting | Value | Impact |
|---------|-------|--------|
| CHUNK_SIZE | 150 | Larger = broader context, fewer chunks |
| OVERLAP | 30 | Larger = smoother transitions, more chunks |
| Total Chunks | ~500-1000 | Depends on PDF size |

### **Real Example:**

For ISO 27001 PDF (50 pages, ~15,000 words):
```
CHUNK_SIZE = 150 words
OVERLAP = 30 words
STEP = 120 words

Chunks created: ~125 chunks
Each chunk: ~150 words (1 paragraph)
Embeddings: 125 × 384-dimensional vectors
```

---

## 7.3 Component 3: Vector Store (FAISS)

### **File**: `rag/vector_store.py`

### **What it does:**
Creates and manages a searchable database of document embeddings.

### **Concept: Embeddings**

An embedding is a numerical representation of text:

```
Text: "ISO 27001 is an international security standard"
           ↓ (Sentence Transformer model)
Embedding: [0.234, -0.812, 0.456, 0.123, ..., 0.789]
           └─ 384 numbers representing semantic meaning

Text: "ISO/IEC 27001 covers information security management"
           ↓
Embedding: [0.245, -0.805, 0.450, 0.118, ..., 0.791]
           └─ Similar text = similar embedding

Similarity Score: cos(v1, v2) = 0.987 (very similar!)
```

### **FAISS (Facebook AI Similarity Search)**

```
Problem: Searching 1,000 embeddings for similar ones = 1,000 comparisons

Solution: FAISS uses optimized algorithms
  ├─ IndexFlatL2: Euclidean distance (simple, accurate)
  ├─ Speed: < 1ms for 1,000 embeddings
  └─ Scalable to millions of embeddings
```

### **Implementation:**

```python
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

class FAISSRetrieval:
    def __init__(self, documents):
        # 1. Load embedding model
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.documents = documents
        
        # 2. Convert all documents to embeddings
        #    Each embedding: 384-dimensional vector
        self.embeddings = self.model.encode(documents)
        # Shape: (num_documents, 384)
        
        # 3. Create FAISS index
        dim = self.embeddings.shape[1]  # 384
        self.index = faiss.IndexFlatL2(dim)
        
        # 4. Add all embeddings to index
        self.index.add(np.array(self.embeddings).astype("float32"))
    
    def retrieve(self, query, top_k=5):
        # 1. Convert query to embedding
        query_embedding = self.model.encode([query]).astype("float32")
        
        # 2. Search FAISS index
        _, indices = self.index.search(query_embedding, top_k)
        # Returns indices of top_k most similar documents
        
        # 3. Get actual documents
        results = [self.documents[i] for i in indices[0]]
        return results
```

### **Search Process:**

```
Query: "What is ISO 27001?"
    ↓
Convert to embedding: [0.230, -0.810, 0.452, ..., 0.790]
    ↓
Search FAISS index (Compare with all 125 embeddings)
    ├─ Document 5: similarity = 0.985 ✓ (Most similar)
    ├─ Document 12: similarity = 0.976 ✓
    ├─ Document 18: similarity = 0.965 ✓
    ├─ Document 3: similarity = 0.821
    └─ Document 45: similarity = 0.234
    ↓
Return top-3: [Document 5, Document 12, Document 18]
    ↓
Extract original text of these documents
    ↓
Use as context for LLM
```

### **Embedding Model: all-MiniLM-L6-v2**

```
Why this model?
  ✓ Lightweight: Only 22 MB
  ✓ Fast: No GPU needed
  ✓ Good quality: Semantic similarity understanding
  ✓384-dimensional: Good balance of speed/accuracy
  ✓ Free: No API costs
  ✓ Open source: Can be self-hosted

Alternatives:
  - all-mpnet-base-v2: Better quality, 420 MB, slower
  - distiluse-base-multilingual: Supports 50+ languages
  - all-roberta-large-v1: High quality, requires GPU
```

---

## 7.4 Component 4: LLM Integration (Ollama)

### **File**: `rag/llm.py`

### **What it does:**
Interfaces with Ollama to generate answers using LLMs.

### **Ollama: Local LLM Service**

```
Traditional LLM (Cloud):
  Question → API Request to OpenAI/Azure/etc → Pay per token → Answer
  Concerns: Privacy, Cost, Latency, Dependency

Ollama (Local):
  Question → Local LLM Process → Instant Response → Free
  Benefits: Private, No costs, Fast, Offline-capable
```

### **Ollama Architecture:**

```
┌─────────────────────────────────────┐
│     Ollama Local LLM Server         │
│   (running on http://127.0.0.1:11434)
│                                     │
│  Models:                            │
│  ├─ mistral (7B, 2.6 GB)           │
│  ├─ llama2 (7B, 3.8 GB)            │
│  ├─ neural-chat (7B, 4.1 GB)       │
│  └─ orca-mini (3B, 1.9 GB)         │
│                                     │
└─────────────────────────────────────┘
         ↑
  HTTP Requests from
  Your Application
```

### **Ollama API**

```
Endpoint: POST /api/generate

Request:
{
  "model": "mistral",
  "prompt": "What is ISO 27001?",
  "stream": false,
  "num_predict": 200
}

Response:
{
  "model": "mistral",
  "created_at": "2024-01-15T10:30:45Z",
  "response": "ISO 27001 is an international information security management...",
  "done": true,
  "context": [...],
  "total_duration": 2500000000,
  "load_duration": 500000000,
  "prompt_eval_count": 45,
  "eval_count": 120,
  "eval_duration": 1500000000
}
```

### **Implementation:**

```python
import requests

class OllamaLLM:
    def __init__(self, model_name, base_url, timeout=30):
        self.model_name = model_name
        self.base_url = base_url
        self.timeout = timeout
    
    def generate(self, prompt: str):
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "num_predict": 200  # Limit output length
        }
        
        response = requests.post(
            f"{self.base_url}/api/generate",
            json=payload,
            timeout=self.timeout
        )
        
        response.raise_for_status()
        return response.json().get("response", "")
```

### **Model Comparison:**

| Model | Size | Speed | Quality | RAM |
|-------|------|-------|---------|-----|
| orca-mini | 3B | ⚡ Fast | ⭐⭐ Good | 2 GB |
| mistral | 7B | ⚡ Medium | ⭐⭐⭐ Great | 4 GB |
| neural-chat | 7B | ⚡ Medium | ⭐⭐⭐ Great | 4 GB |
| llama2 | 7B | 🐢 Slow | ⭐⭐⭐⭐ Excellent | 4 GB |
| llama2-13b | 13B | 🐢 Very Slow | ⭐⭐⭐⭐⭐ Best | 8 GB |

### **Configuration Options:**

```python
# Temperature (0-1): Randomness of responses
# Lower: More predictable, focused
# Higher: More creative, varied
num_ctx = 4096          # Context window size
temperature = 0.7       # Balanced creativity

# Top-P (0-1): Nucleus sampling
# Lower: More focused
# Higher: More diverse
top_p = 0.9

# Top-K: Consider only top-K most likely tokens
top_k = 40

# num_predict: Max tokens to generate
num_predict = 200       # Limit output length
```

---

## 7.5 Component 5: RAG Pipeline

### **File**: `rag/pipeline.py`

### **What it does:**
Orchestrates the entire RAG workflow.

### **Pipeline Flow:**

```
User Query: "What is ISO 27001?"
    ↓
┌───────────────────────────────────────────┐
│  STEP 1: RETRIEVE                         │
│  ├─ Convert query to embedding            │
│ ├─ Search FAISS for similar chunks       │
│  └─ Retrieve top-3 matching documents    │
└───────────────────────────────────────────┘
    ↓
Retrieved Contexts:
[
  "ISO 27001 is an international standard...",
  "It covers management of information security...",
  "Organizations implement ISO 27001 to protect..."
]
    ↓
┌───────────────────────────────────────────┐
│  STEP 2: AUGMENT (Build Prompt)          │
│  Create prompt combining contexts + query│
└───────────────────────────────────────────┘
    ↓
Augmented Prompt:
"""
Answer the question using ONLY the context below.
Be concise (1-2 sentences).

Context:
ISO 27001 is an international standard...
It covers management of information security...
Organizations implement ISO 27001 to protect...

Question: What is ISO 27001?
Answer:
"""
    ↓
┌───────────────────────────────────────────┐
│  STEP 3: GENERATE                         │
│  ├─ Send prompt to Ollama LLM             │
│  └─ LLM generates answer based on context│
└───────────────────────────────────────────┘
    ↓
Generated Answer:
"ISO 27001 is an international standard for
information security management systems that
helps organizations manage the security of
assets such as financial information and
intellectual property."
    ↓
┌───────────────────────────────────────────┐
│  STEP 4: RETURN                           │
│  ├─ Answer to user                        │
│  └─ Show source contexts                 │
└───────────────────────────────────────────┘
    ↓
Response to User:
{
  "answer": "ISO 27001 is an international...",
  "contexts": ["ISO 27001 is...", "It covers...", "Organizations..."]
}
```

### **Implementation:**

```python
class SimpleRAGPipeline:
    def __init__(self, retrieval, llm, top_k=3):
        self.retrieval = retrieval
        self.llm = llm
        self.top_k = top_k
    
    def run(self, query: str):
        # STEP 1: RETRIEVE
        docs = self.retrieval.retrieve(query, self.top_k)
        
        # STEP 2: AUGMENT
        context = "\n---\n".join(docs)
        prompt = f"""Answer using ONLY this context:

Context:
{context}

Question: {query}
Answer:"""
        
        # STEP 3: GENERATE
        answer = self.llm.generate(prompt)
        
        # STEP 4: RETURN
        return Answer(answer, docs)
```

---

# 8. How It Works: The RAG Pipeline

## 8.1 End-to-End Example: ISO 27001 Question

### **Scenario:**
User opens web application and asks: "What are the main objectives of ISO 27001?"

### **Complete Request-Response Flow:**

```
┌─────────────────────────────────────────────────────────────┐
│ 1. USER INTERACTION                                         │
│    User types: "What are the main objectives of ISO 27001?"│
│    Clicks: "Send" or presses Enter                         │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│ 2. FRONTEND (Browser/index.html)                            │
│    ├─ Display user message in chat                          │
│    ├─ Show "typing..." indicator                            │
│    └─ Send HTTP POST to http://localhost:5000/api/ask      │
│       POST body:                                            │
│       {                                                     │
│         "question": "What are the main objectives..."       │
│       }                                                     │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│ 3. FLASK PROXY (app_flask.py)                              │
│    ├─ Receive POST request at /api/ask                     │
│    ├─ Forward to FastAPI backend                           │
│    └─ GET http://127.0.0.1:8000/ask                        │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│ 4. FASTAPI BACKEND (app.py)                                │
│    ├─ Route: POST /ask                                     │
│    ├─ Receive question JSON                                │
│    └─ Check if pipeline initialized                        │
│        └─ If first request: initialize (5-10 seconds)      │
│        └─ Else: use cached pipeline (instant)              │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│ 5. PIPELINE INITIALIZATION (if first request)              │
│                                                             │
│    Step 1: LOAD PDF                                        │
│    ├─ File: example/iso27001.pdf (configured in config.py) │
│    ├─ Library: PyPDF                                       │
│    └─ Result: Full text of all 50 pages                    │
│    Time: ~1 second                                         │
│                                                             │
│    Step 2: CHUNK TEXT                                      │
│    ├─ Size: 150 words per chunk                            │
│    ├─ Overlap: 30 words                                    │
│    └─ Result: ~120 chunks extracted                        │
│    Time: <0.1 seconds                                      │
│                                                             │
│    Step 3: CREATE EMBEDDINGS                               │
│    ├─ Model: all-MiniLM-L6-v2                             │
│    ├─ Convert: Each chunk → 384-dim vector                │
│    └─ Result: 120 embeddings created                       │
│    Time: ~2 seconds                                        │
│                                                             │
│    Step 4: BUILD FAISS INDEX                               │
│    ├─ Algorithm: IndexFlatL2                              │
│    ├─ Store: All 120 embeddings                            │
│    └─ Ready: For fast similarity search                    │
│    Time: <0.1 seconds                                      │
│                                                             │
│    Step 5: INITIALIZE LLM                                  │
│    ├─ Model: mistral (from config.py)                      │
│    ├─ URL: http://127.0.0.1:11434                         │
│    └─ Status: Ready for generation                         │
│    Time: <0.1 seconds                                      │
│                                                             │
│    Total initialization: ~3-5 seconds                      │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│ 6. RAG PIPELINE EXECUTION (for every query)               │
│                                                             │
│    PHASE 1: RETRIEVE RELEVANT CONTEXTS                      │
│    ├─ Input: "What are the main objectives of ISO 27001?"  │
│    │                                                        │
│    ├─ Step 1: Embed the question                           │
│    │   └─ Query embedding: [0.234, -0.812, ..., 0.789]   │
│    │   └─ Dimensions: 384                                  │
│    │                                                        │
│    ├─ Step 2: Search FAISS index                           │
│    │   ├─ Compare query with all 120 chunk embeddings      │
│    │   ├─ Calculate distances (L2 Euclidean)              │
│    │   └─ Sort by distance                                 │
│    │                                                        │
│    ├─ Step 3: Get top-3 most similar chunks               │
│    │   ├─ Chunk #45: "ISO 27001 objectives include..."   │
│    │   ├─ Chunk #47: "Main goals are to ensure..."        │
│    │   └─ Chunk #52: "Organizations implementing ISO..."  │
│    │                                                        │
│    └─ Result: 3 relevant document chunks                   │
│    Time: <0.1 seconds                                      │
│                                                             │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│ 7. AUGMENTATION PHASE (Build Context-Augmented Prompt)     │
│                                                             │
│    Take retrieved contexts and build prompt:                │
│                                                             │
│    Prompt Template:                                         │
│    """                                                      │
│    Answer the question using ONLY the context below.        │
│    Be concise (1-2 sentences).                              │
│                                                             │
│    Context:                                                 │
│    [Context 1 from Chunk #45]                              │
│    ---                                                      │
│    [Context 2 from Chunk #47]                              │
│    ---                                                      │
│    [Context 3 from Chunk #52]                              │
│                                                             │
│    Question: What are the main objectives of ISO 27001?    │
│    Answer:                                                  │
│    """                                                      │
│                                                             │
│    Key points:                                              │
│    • Instruction: Tell LLM to use ONLY provided context    │
│    • Contexts: 3 most relevant document chunks              │
│    • Question: User's original question                     │
│    • Answer: Empty - LLM will fill this                     │
│                                                             │
│    Why augmentation?                                        │
│    ✓ Grounds answer in your documents                       │
│    ✓ Reduces hallucinations                                 │
│    ✓ Ensures factual accuracy                              │
│    ✓ Provides source material                               │
│                                                             │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│ 8. GENERATION PHASE (LLM Answer)                           │
│                                                             │
│    Action: Send augmented prompt to Ollama                  │
│    ├─ HTTP POST to http://127.0.0.1:11434/api/generate   │
│    ├─ Model: mistral                                       │
│    ├─ Prompt: (from step 7)                                │
│    ├─ Timeout: 30 seconds                                  │
│    └─ num_predict: 200 tokens max                          │
│                                                             │
│    Ollama Processing:                                       │
│    ├─ Load model to memory                                  │
│    ├─ Tokenize prompt                                       │
│    ├─ Generate tokens one by one                            │
│    ├─ Stop at 200 tokens OR end of sentence                │
│    └─ Return complete response                              │
│                                                             │
│    Response:                                                │
│    {                                                        │
│      "model": "mistral",                                    │
│      "response": "ISO 27001 objectives include establishing│
│      an information security management system (ISMS),      │
│      protecting confidentiality and integrity of sensitive │
│      information, and demonstrating compliance with        │
│      security controls and regulations.",                  │
│      "done": true                                          │
│    }                                                        │
│                                                             │
│    LLM Think Process:                                       │
│    ├─ Reads contexts (what ISO 27001 is)                   │
│    ├─ Understands question (what are objectives)           │
│    ├─ Synthesizes answer (combine context + logic)         │
│    └─ Outputs relevant response                             │
│                                                             │
│    Time: 2-10 seconds (depends on model & system)          │
│                                                             │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│ 9. RESPONSE ASSEMBLY (Backend)                              │
│                                                             │
│    Create JSON response:                                    │
│    {                                                        │
│      "answer": "ISO 27001 objectives include establishing│
│      an information security management system (ISMS),     │
│      protecting confidentiality and integrity...",         │
│      "contexts": [                                          │
│        "ISO 27001 objectives include establishing...",     │
│        "Main goals are to ensure information security...", │
│        "Organizations implementing ISO 27001 must..."      │
│      ]                                                      │
│    }                                                        │
│                                                             │
│    Include:                                                 │
│    ├─ Final answer from LLM                                │
│    └─ Source documents (for transparency)                  │
│                                                             │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│ 10. RESPONSE ROUTING (Flask Proxy)                          │
│                                                             │
│    ├─ Receive response from FastAPI                         │
│    ├─ Forward to browser                                    │
│    └─ HTTP 200 OK with JSON body                            │
│                                                             │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│ 11. FRONTEND DISPLAY (Browser/JavaScript)                  │
│                                                             │
│    ├─ Parse JSON response                                   │
│    ├─ Display answer in chat bubble                         │
│    ├─ Show "Sources:" section                               │
│    ├─ Display retrieved contexts                            │
│    ├─ Hide "typing..." indicator                            │
│    └─ Ready for next question                               │
│                                                             │
│    Chat Display:                                            │
│    ┌─────────────────────────────────────────────────┐    │
│    │ You: What are the main objectives of ISO 27001? │    │
│    │                                                 │    │
│    │ Bot: ISO 27001 objectives include establishing  │    │
│    │      an information security management system  │    │
│    │      (ISMS), protecting confidentiality and     │    │
│    │      integrity...                               │    │
│    │                                                 │    │
│    │ 📚 Sources:                                     │    │
│    │ [1] ISO 27001 objectives include establishing...│    │
│    │ [2] Main goals are to ensure...                 │    │
│    │ [3] Organizations implementing ISO 27001...     │    │
│    └─────────────────────────────────────────────────┘    │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### **Timing Summary:**

| Phase | Time | Notes |
|-------|------|-------|
| Network latency | ~50ms | Browser to server |
| Retrieve | ~100ms | Vector search in FAISS |
| Augment | ~10ms | Build prompt |
| Generate | 2-10s | LLM thinking time |
| Return | ~50ms | Response to browser |
| **Total (first request)** | **~10-15s** | Includes initialization |
| **Total (subsequent)** | **~2-11s** | No initialization |

---

# 9. Configuration Guide

## 9.1 Configuration Methods

### **Method 1: Environment Variables (.env file)**

Create `.env` file in project root:
```env
MODEL_NAME=mistral
OLLAMA_BASE_URL=http://127.0.0.1:11434
CHUNK_SIZE=150
CHUNK_OVERLAP=30
TOP_K=3
```

Advantages:
- ✅ Easy to change
- ✅ Doesn't modify code
- ✅ Secure (not in git)
- ✅ Environment-specific

### **Method 2: Direct in Code (rag/config.py)**

Edit `rag/config.py`:
```python
MODEL_NAME = "mistral"
OLLAMA_BASE_URL = "http://127.0.0.1:11434"
CHUNK_SIZE = 150
CHUNK_OVERLAP = 30
TOP_K = 3
```

Advantages:
- ✅ Quick testing
- ✅ No file dependency
- ✗ Changes require code edit

### **Method 3: Command Line**

```bash
# Override settings temporarily
MODEL_NAME=llama2 CHUNK_SIZE=200 python app.py
```

## 9.2 Configuration Parameters

### **MODEL_NAME**
Which LLM to use with Ollama.

```
Available Models:
  • mistral       = Fast, good quality (7B, 2.6 GB) ✓ Recommended
  • llama2        = Best quality, slower (7B, 3.8 GB)
  • neural-chat   = Good for conversations (7B, 4.1 GB)
  • orca-mini     = Very fast, smaller (3B, 1.9 GB)

Download: ollama pull <model-name>
List: ollama list

Example Configuration:
  For speed: MODEL_NAME=orca-mini
  For quality: MODEL_NAME=llama2
  Balanced: MODEL_NAME=mistral (default)
```

### **OLLAMA_BASE_URL**
Where the Ollama service is running.

```
Default: http://127.0.0.1:11434

Custom configurations:
  Local machine: http://127.0.0.1:11434
  Different port: http://127.0.0.1:11435 (if Ollama on 11435)
  Remote server: http://192.168.1.100:11434
  Docker: http://ollama:11434 (if both in Docker)
  
Important: 
  • Ollama must be running
  • Firewall must allow connection
  • No HTTPS required for localhost
```

### **CHUNK_SIZE**
Number of words per chunk.

```
Impact: Chunk_size → Context size → Answer quality

Small chunks (100):
  ├─ Pros: Very specific contexts, more chunks
  ├─ Cons: Might miss broader context
  └─ Good for: Factual, precise documents

Medium chunks (150):
  ├─ Pros: Balanced (default)
  ├─ Cons: None
  └─ Good for: Most use cases

Large chunks (200+):
  ├─ Pros: Broader context, fewer chunks
  ├─ Cons: Less relevant per chunk
  └─ Good for: Narrative, story-like documents

Calculation:
  Total words in PDF / Chunk_size = Approximate number of chunks
  Example: 15,000 words / 150 words = ~100 chunks
```

### **CHUNK_OVERLAP**
Words to repeat between chunks.

```
Without overlap:
  Chunk 1: "word1 word2 ... word150"
  Chunk 2: "word151 word152 ... word300"
  Risk: Lose information at boundary

With overlap:
  Chunk 1: "word1 word2 ... word150"
  Chunk 2: "word121 word122 ... word270"  (overlap = 30)
  Benefit: Information preserved at boundaries

Recommended: 20-50% of chunk_size
  • CHUNK_SIZE=150, OVERLAP=30 (20%)
  • CHUNK_SIZE=200, OVERLAP=50 (25%)
  • CHUNK_SIZE=100, OVERLAP=20 (20%)

Effect on chunks:
  With overlap, total chunks increases but answers improve
```

### **TOP_K**
How many document chunks to retrieve per question.

```
TOP_K = 1:
  └─ Retrieve only 1 most similar chunk
  ├─ Pros: Fast, focused
  ├─ Cons: Might miss important context
  └─ Use case: Simple factual questions

TOP_K = 3 (default):
  ├─ Retrieve 3 most similar chunks
  ├─ Pros: Good balance of quality and speed
  ├─ Cons: None
  └─ Use case: General purpose

TOP_K = 5:
  ├─ Retrieve 5 most similar chunks
  ├─ Pros: More comprehensive answers
  ├─ Cons: Slower, might include noise
  └─ Use case: Complex questions, comprehensive answers

TOP_K = 10+:
  ├─ Retrieve many chunks
  ├─ Pros: Maximum context
  ├─ Cons: Slow, LLM gets confused with too much info
  ├─ Risk: Token limit (LLM input too large)
  └─ Not recommended

Sweet spot: 3-5 for most use cases
```

## 9.3 Performance Tuning

### **Goal: Speed Up Responses**

```
Current Performance:
  ├─ First request: 10-15 seconds
  └─ Subsequent requests: 3-5 seconds

Optimization Strategy:

1. Reduce chunk processing:
   MODEL_NAME=orca-mini          # Fastest model
   CHUNK_SIZE=100                # Fewer chunks to process
   TOP_K=1                        # Less retrieval time
   
   Result: 2-4 seconds per query

2. Reduce context size:
   TOP_K=1                        # Minimal context
   CHUNK_SIZE=100                # Smaller chunks
   
   Risk: Answers might be less accurate

3. Use faster GPU:
   • Install GPU CUDA drivers
   • FAISS-GPU for vector search
   • Ollama GPU acceleration
   
   Result: 50-70% faster
```

### **Goal: Improve Answer Quality**

```
Optimization Strategy:

1. Better model:
   MODEL_NAME=llama2             # Better quality
   
   Trade-off: Slower (5-15 seconds)

2. More context:
   TOP_K=5                        # More sources
   CHUNK_SIZE=200                # More context per chunk
   CHUNK_OVERLAP=50              # Better transitions
   
   Trade-off: Slower

3. Better embeddings:
   # Edit vector_store.py to use:
   # "all-mpnet-base-v2" instead of "all-MiniLM-L6-v2"
   
   Trade-off: 200 MB larger model, 50% slower

4. Combine strategies:
   MODEL_NAME=llama2
   TOP_K=5
   CHUNK_SIZE=150
   
   Result: Better answers, slower responses
```

---

# 10. Feature Implementation Guide

## 10.1 Available Extensions

### **Feature 1: Add Support for Multiple Documents**

**Current State:** Only works with one PDF at a time

**Implementation:**

```python
# 1. Create documents manager
class DocumentManager:
    def __init__(self):
        self.documents = {}  # doc_id -> text
        self.index = None
    
    def add_document(self, doc_id, pdf_path):
        reader = PDFReader(pdf_path)
        content = reader.read()[0]
        chunks = text2chunk(content, CHUNK_SIZE, CHUNK_OVERLAP)
        self.documents[doc_id] = chunks
        self._rebuild_index()
    
    def _rebuild_index(self):
        # Combine all chunks from all documents
        all_chunks = []
        for chunks in self.documents.values():
            all_chunks.extend(chunks)
        self.index = FAISSRetrieval(all_chunks)

# 2. Update API endpoint
@app.post("/add_document")
async def add_document(doc_id: str, file: UploadFile = File(...)):
    # Save uploaded file
    # Add to manager
    # Return success

# 3. Update UI
# Add file upload button
# Show list of loaded documents
# Add document selector filter
```

**Estimated Effort:** 4-6 hours
**Complexity:** Medium

### **Feature 2: Conversation History**

**Current State:** Stateless - each question independent

**Implementation:**

```python
# 1. Store conversation context
class ConversationHistory:
    def __init__(self):
        self.history = []  # List of (question, answer) tuples
    
    def add(self, question, answer):
        self.history.append({
            "question": question,
            "answer": answer,
            "timestamp": datetime.now()
        })
    
    def get_context(self, max_messages=5):
        # Return last N messages for context
        return self.history[-max_messages:]

# 2. Update LLM prompt
# Include previous conversation
# Give LLM context of what was discussed

# 3. Frontend changes
# Display conversation history
# Add "Clear Chat" button
# Show conversation timeline
```

**Estimated Effort:** 3-4 hours
**Complexity:** Easy-Medium

### **Feature 3: Query Rewriting & Expansion**

**Current State:** Sends user query directly to retrieval

**Implementation:**

```python
# 1. Add query enhancement
def enhance_query(query, llm):
    """
    Convert user query into better search terms
    Example:
      User: "What is it?"
      Rewritten: "What is ISO 27001? Define ISO 27001."
    """
    prompt = f"""Expand this query into 3 variations for better document retrieval:
    Query: {query}
    
    Return only the 3 queries, one per line."""
    
    expanded = llm.generate(prompt)
    queries = expanded.split("\n")
    return queries

# 2. Multi-query retrieval
def retrieve_with_expansion(query, retrieval, llm, top_k=3):
    queries = enhance_query(query, llm)
    all_results = []
    for q in queries:
        results = retrieval.retrieve(q, top_k)
        all_results.extend(results)
    
    # Deduplicate and rank
    unique_results = list(set(all_results))
    return unique_results[:top_k*2]

# 3. Update pipeline
# Use enhanced retrieval instead of simple
```

**Estimated Effort:** 4-5 hours
**Complexity:** Medium

### **Feature 4: Answer Evaluation & Sources**

**Current State:** Shows answer and raw sources

**Implementation:**

```python
# 1. Rank sources by relevance
from sklearn.metrics.pairwise import cosine_similarity

def rank_sources(query, contexts, embeddings):
    query_emb = SentenceTransformer("...").encode(query)
    scores = []
    for ctx_emb in embeddings:
        score = cosine_similarity([query_emb], [ctx_emb])[0][0]
        scores.append(score)
    
    ranked = sorted(zip(contexts, scores), key=lambda x: x[1], reverse=True)
    return ranked

# 2. Extract direct answer from contexts
def extract_answer_span(question, contexts):
    """Find the most relevant sentence in contexts"""
    # Use keyword matching or semantic search
    pass

# 3. Confidence scoring
def calculate_confidence(answer, contexts):
    """How confident are we in this answer?"""
    return score  # 0.0-1.0
```

**Estimated Effort:** 5-6 hours
**Complexity:** Medium-Hard

### **Feature 5: Web UI Improvements**

**Current State:** Basic chat interface

**Enhancements:**

```html
<!-- 1. Add settings panel -->
<div id="settings-panel">
  <label>Model: <select id="model-select">
    <option>mistral</option>
    <option>llama2</option>
  </select></label>
  <label>Top-K: <input type="number" id="topk-input" value="3"></label>
</div>

<!-- 2. Add typing indicator animation -->
<div class="typing-indicator">
  <span></span><span></span><span></span>
</div>

<!-- 3. Add copy buttons on sources -->
<div class="source-item">
  <button onclick="copyToClipboard(this)">📋 Copy</button>
  <p>Source text...</p>
</div>

<!-- 4. Add dark mode toggle -->
<button id="theme-toggle">🌙 Dark Mode</button>

<!-- 5. Add download transcript -->
<button onclick="downloadConversation()">💾 Download</button>
```

**Estimated Effort:** 2-3 hours
**Complexity:** Easy

### **Feature 6: RAG with Web Search**

**Current State:** Only uses documents

**Implementation:**

```python
# 1. Add web search fallback
def rag_with_fallback(query, retrieval, llm, web_search=True):
    # First try document retrieval
    docs = retrieval.retrieve(query, top_k=3)
    
    if not docs or confidence_low:
        if web_search:
            # Search web for more context
            web_results = google_search(query, num_results=3)
            docs.extend(web_results)
    
    # Generate answer with combined sources
    answer = generate_answer(docs, llm)
    return answer

# 2. Add web search library
# pip install google-search-results
# or DuckDuckGo search

# 3. Update UI
# Add "Search Web" toggle
# Show web sources separately
```

**Estimated Effort:** 4-6 hours
**Complexity:** Medium

### **Feature 7: Advanced Retrieval Methods**

**Current State:** Basic FAISS similarity

**Enhancements:**

```python
# 1. Hybrid retrieval (semantic + keyword)
from rank_bm25 import BM25Okapi

class HybridRetrieval:
    def __init__(self, documents):
        self.semantic = FAISSRetrieval(documents)
        self.bm25 = BM25Okapi([d.split() for d in documents])
    
    def retrieve(self, query, top_k=3):
        # Semantic search
        semantic_results = self.semantic.retrieve(query, top_k)
        
        # Keyword search (BM25)
        scores = self.bm25.get_scores(query.split())
        keyword_results = [self.documents[i] for i in np.argsort(scores)[-top_k:]]
        
        # Combine
        return combine_results(semantic_results, keyword_results)

# 2. Reranking
from sentence_transformers import CrossEncoder

class RankedRetrieval:
    def __init__(self):
        self.ranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    
    def retrieve(self, query, candidates, top_k=3):
        # Rank candidates by relevance to query
        scores = self.ranker.predict([[query, doc] for doc in candidates])
        ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
        return [doc for doc, score in ranked[:top_k]]

# 3. Maximal Marginal Relevance (MMR)
# Reduce redundancy in retrieved documents
```

**Estimated Effort:** 6-8 hours
**Complexity:** Hard

### **Feature 8: Authentication & Multi-User**

**Current State:** No authentication

**Implementation:**

```python
# 1. Add user authentication
from fastapi.security import HTTPBearer, HTTPAuthCredentials

security = HTTPBearer()

@app.post("/ask")
async def ask(question: str, credentials: HTTPAuthCredentials = Depends(security)):
    user_id = verify_token(credentials.credentials)
    # Process question for specific user
    return response

# 2. Database for user data
# SQLAlchemy + SQLite/PostgreSQL
# Store: user_id, question, answer, timestamp

# 3. Session management
# JWT tokens for auth
# Redis for session storage

# 4. Per-user documents
# Each user can upload own PDFs
# Private RAG system per user
```

**Estimated Effort:** 8-12 hours
**Complexity:** Hard

---

## 10.2 Implementation Template for New Features

### **Step 1: Design**
```
What: What feature to add?
Why: Why is it needed?
How: Architecture overview
When: When should it run?
Cost: Performance impact?
```

### **Step 2: Implement Core Logic**
```python
# Create new module
# rag/new_feature.py

class NewFeature:
    def __init__(self):
        pass
    
    def process(self, input):
        # Implementation
        return output
```

### **Step 3: Integrate with Pipeline**
```python
# Modify app.py or pipeline.py
# Add new_feature to workflow
```

### **Step 4: Update API**
```python
# Add endpoint if needed
@app.post("/new_endpoint")
async def new_endpoint(data: NewModel):
    return response
```

### **Step 5: Update UI**
```html
<!-- Add UI components -->
<!-- Add JavaScript event handlers -->
```

### **Step 6: Test**
```python
# Test new feature
# Test integration
# Test performance impact
```

---

# 11. Troubleshooting Guide

## 11.1 Common Issues and Solutions

### **Issue 1: "Connection refused" / "Cannot connect to Ollama"**

**Symptoms:**
```
Error: Failed to connect to http://127.0.0.1:11434
ConnectionError: Connection refused
```

**Causes:**
- Ollama service not running
- Wrong URL in config
- Ollama crashed

**Solutions:**

```bash
# 1. Check if Ollama is running
curl http://127.0.0.1:11434/api/tags

# 2. If not running, start it
ollama serve

# 3. Verify URL in config
# Check .env or rag/config.py
# Should be: http://127.0.0.1:11434

# 4. Check port (Windows)
netstat -ano | find "11434"

# 5. Restart Ollama
# Windows: Restart from Services
# Or: taskkill /F /IM ollama.exe
# Then: ollama serve
```

---

### **Issue 2: "Model not found" / "unknown model"**

**Symptoms:**
```
Error: unknown model 'llama2'
Model 'mistral' not found
```

**Causes:**
- Model not downloaded
- Model name misspelled
- Ollama cache corrupted

**Solutions:**

```bash
# 1. List available models
ollama list

# 2. Download missing model
ollama pull mistral
ollama pull llama2
ollama pull neural-chat

# 3. Check model name spelling
# mistral (not mistral-ai, mistral7b, etc.)
# llama2 (not llama, llama-2, etc.)

# 4. Clear Ollama cache (if corrupted)
# Windows: %USERPROFILE%\.ollama
# Mac: ~/.ollama
# Linux: ~/.ollama
# Delete and re-download model
```

---

### **Issue 3: "Module not found" / "ImportError"**

**Symptoms:**
```
ModuleNotFoundError: No module named 'faiss'
ImportError: No module named 'sentence_transformers'
```

**Causes:**
- Dependencies not installed
- Wrong virtual environment
- Virtual environment broken

**Solutions:**

```bash
# 1. Verify virtual environment activated
# Prompt should show (rfl)
# If not:
.\rfl\Scripts\Activate.ps1          # Windows PowerShell
rfl\Scripts\activate.bat             # Windows CMD

# 2. Install/reinstall dependencies
pip install -r requirements.txt

# 3. Check specific module
pip list | find "faiss"
pip list | find "sentence-transformers"

# 4. Reinstall virtual environment
deactivate                           # Deactivate current
rmdir /s /q rfl                      # Remove old venv
python -m venv rfl                   # Create new
.\rfl\Scripts\Activate.ps1          # Activate
pip install -r requirements.txt      # Install deps
```

---

### **Issue 4: "PDF file not found"**

**Symptoms:**
```
FileNotFoundError: [Errno 2] No such file or directory: 'example/iso27001.pdf'
```

**Causes:**
- PDF doesn't exist at configured path
- Wrong path in config
- File deleted

**Solutions:**

```bash
# 1. Verify PDF exists
ls -la example/iso27001.pdf          # Check file

# 2. Check config path
# rag/config.py or .env
# Should match actual file location

# 3. Provide absolute path if needed
PDF_PATH=C:\Users\VISRUTHI\Documents\GenAI\RAG_FAISS_LLAMA\example\iso27001.pdf

# 4. Restore example PDF
# Download from repository or
# Use any PDF for testing
```

---

### **Issue 5: "Port already in use"**

**Symptoms:**
```
Address already in use
ERROR: address already in use (port 8000)
Address already in use: ('127.0.0.1', 5000)
```

**Causes:**
- FastAPI or Flask already running
- Previous process not stopped
- Another application using port

**Solutions:**

```bash
# 1. Kill Python processes
taskkill /F /IM python.exe

# 2. Use different port
# Terminal 1 (FastAPI on port 8001)
python -m uvicorn app:app --port 8001

# Terminal 2 (Flask)
# Edit app_flask.py: API_URL = "http://127.0.0.1:8001"
python app_flask.py --port 5001

# 3. Find process using port
netstat -ano | find "8000"
taskkill /PID <PID> /F

# 4. Restart from scratch
# Close all terminals
# Start fresh
```

---

### **Issue 6: "Out of Memory" / "GPU memory full"**

**Symptoms:**
```
RuntimeError: CUDA out of memory
MemoryError: Unable to allocate... GB
```

**Causes:**
- Model too large for available RAM
- Too many embeddings at once
- Memory leak (long running)

**Solutions:**

```bash
# 1. Use smaller model
MODEL_NAME=orca-mini          # 3B model, uses less RAM

# 2. Reduce chunk size
CHUNK_SIZE=100                # Smaller chunks = less memory

# 3. Check available memory
# Windows: Check Task Manager
# Linux: free -h

# 4. Reduce context
TOP_K=1                        # Retrieve less

# 5. Restart services
# Memory leaks accumulate over time
# Restart Ollama and backend

# 6. Switch to CPU-only FAISS
# Already using CPU version (faiss-cpu)
# GPU version not needed for vector search

# 7. Monitor memory
import psutil
print(psutil.virtual_memory())
```

---

### **Issue 7: "LLM timeout" / "Request timed out"**

**Symptoms:**
```
requests.exceptions.Timeout: HTTPConnectionPool timeout
TimeoutError after 30s
```

**Causes:**
- LLM generating too long
- System slow (HDD vs SSD)
- Network issues

**Solutions:**

```bash
# 1. Increase timeout
# Edit rag/llm.py
timeout=60  # Increase from 30s

# 2. Use faster model
MODEL_NAME=orca-mini          # Very fast

# 3. Reduce output length
# Edit rag/llm.py
"num_predict": 100             # Reduce from 200

# 4. Check system performance
# Task Manager → Performance
# CPU usage, Memory, Disk

# 5. Enable GPU acceleration
# Install CUDA
# Let Ollama use GPU

# 6. Reduce Ollama load
# Stop other applications
# Close browser tabs
```

---

### **Issue 8: "Empty or incorrect answer"**

**Symptoms:**
```
LLM returns: "I don't know" or garbage text
Answer completely unrelated to question
```

**Causes:**
- Poor context retrieval
- Wrong documents retrieved
- Model not suitable for task

**Solutions:**

```bash
# 1. Improve chunk size
CHUNK_SIZE=200                 # Larger context

# 2. Retrieve more contexts
TOP_K=5                         # More sources

# 3. Use better model
MODEL_NAME=llama2              # Better quality

# 4. Check retrieved contexts
# Add logging to see what's retrieved
# Verify contexts are actually relevant

# 5. Adjust prompt template
# Edit rag/pipeline.py
# Make prompt more specific

# 6. Check PDF quality
# Is text extractable?
# Scanned PDFs won't work

# 7. Debug mode
# Add print statements
# See actual retrieval results
```

---

## 11.2 Debug Mode

Enable detailed logging:

```python
# Edit app.py
import logging
logging.basicConfig(level=logging.DEBUG)

# Edit rag/pipeline.py
print(f"[DEBUG] Query: {query}")
print(f"[DEBUG] Retrieved docs: {docs}")
print(f"[DEBUG] Prompt: {prompt[:200]}...")
print(f"[DEBUG] Answer: {answer}")
```

**Output shows:**
- Input queries
- Retrieved documents
- Generated prompts
- Final answers

---

## 11.3 Performance Debugging

```python
import time

def measure_performance(query):
    start = time.time()
    
    # Step 1: Retrieve
    t1 = time.time()
    docs = retrieval.retrieve(query, top_k=3)
    print(f"Retrieval: {(time.time()-t1)*1000:.0f}ms")
    
    # Step 2: Augment
    t2 = time.time()
    prompt = build_prompt(query, docs)
    print(f"Augmentation: {(time.time()-t2)*1000:.0f}ms")
    
    # Step 3: Generate
    t3 = time.time()
    answer = llm.generate(prompt)
    print(f"Generation: {(time.time()-t3)*1000:.0f}ms")
    
    print(f"Total: {(time.time()-start)*1000:.0f}ms")
```

---

# 12. Advanced Topics

## 12.1 Production Deployment

### **From Local to Production**

```
Development (Current)
├─ Running on localhost
├─ Single PDF
├─ No authentication
├─ No database
└─ Manual startup

Production
├─ Running on server/cloud
├─ Multiple documents
├─ User authentication
├─ Persistent database
├─ Automated deployment
└─ Monitoring & logging
```

### **Docker Containerization**

```dockerfile
# Dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

CMD ["python", "app.py"]
```

**Build and run:**
```bash
docker build -t rag-app .
docker run -p 8000:8000 -p 5000:5000 rag-app
```

---

### **Cloud Deployment (AWS Example)**

```
1. Create EC2 instance
2. Install Python, Ollama
3. Clone repository
4. Configure security groups
5. Use ECS for containers
6. Set up RDS for database
7. Use CloudFront for CDN
8. Enable auto-scaling
```

---

## 12.2 Scaling the System

### **Problem: Slow with large PDFs (1000+ pages)**

**Solution 1: Hierarchical Chunking**
```python
# Split document by sections first
# Then chunk each section
# Faster searching within section
```

**Solution 2: Index Sharding**
```python
# Split embeddings across multiple indices
# Search in parallel
# Combine results
```

**Solution 3: Vector DB (Advanced)**
```python
# Use Pinecone, Weaviate, or Milvus
# Handle millions of vectors
# Built-in scaling
```

---

## 12.3 Advanced RAG Techniques

### **1. Hypothetical Document Embeddings (HyDE)**
```
Instead of: embedding(query)
Do: 
  1. Generate hypothetical relevant document
  2. Embed the document
  3. Use document embedding for search

More semantically similar to documents
```

### **2. ColBERT - Late Interaction**
```
Token-level interactions instead of document-level
Matches individual tokens in query to document
More fine-grained retrieval
```

### **3. Dense Passage Retrieval (DPR)**
```
Separate encoders for question and passage
Better optimization than sentence transformers
Superior retrieval quality
```

---

## 12.4 Monitoring & Observability

```python
# Track metrics
metrics = {
    "total_queries": 0,
    "avg_retrieval_time": 0,
    "avg_generation_time": 0,
    "avg_accuracy": 0,
    "errors": []
}

# Log everything
import logging
logging.basicConfig(
    filename="rag_app.log",
    level=logging.INFO
)

# Monitor endpoints
@app.post("/metrics")
def get_metrics():
    return metrics
```

---

## 12.5 Fine-tuning & Customization

### **Fine-tune Embedding Model**
```python
# Train on domain-specific documents
# Improves semantic understanding of your domain
# Higher relevance scores
```

### **Fine-tune LLM**
```python
# Train Ollama model on your documents
# Better understanding of your content
# More accurate answers
```

---

# Conclusion & Quick Reference

## Quick Command Reference

```bash
# Setup
python -m venv rfl
.\rfl\Scripts\Activate.ps1
pip install -r requirements.txt

# Configuration
# Edit .env or rag/config.py
MODEL_NAME=mistral
CHUNK_SIZE=150
TOP_K=3

# Start Services
ollama serve                     # Terminal 1

# Terminal 2
python -m uvicorn app:app --port 8000 --reload

# Terminal 3
python app_flask.py

# Access
# Browser: http://localhost:5000

# Stop
# Ctrl+C in each terminal
```

---

## Project Structure Recap

```
RAG_FAISS_LLAMA/
├── app.py                  # FastAPI backend
├── app_flask.py           # Flask frontend
├── requirements.txt       # Dependencies
├── rag/
│   ├── config.py         # Settings
│   ├── data_loader.py    # PDF reading
│   ├── text_utils.py     # Chunking
│   ├── vector_store.py   # FAISS
│   ├── llm.py            # Ollama
│   └── pipeline.py       # Orchestration
├── templates/
│   └── index.html        # Web UI
├── example/
│   └── iso27001.pdf      # Sample PDF
└── rfl/                  # Virtual env
```

---

## Key Takeaways

1. **RAG = Retrieval + Augmentation + Generation**
   - Retrieval: Find relevant documents
   - Augmentation: Add context to query
   - Generation: LLM creates answer

2. **Three Main Components**
   - **FAISS**: Fast vector similarity search
   - **Ollama**: Local LLM inference
   - **Flask+FastAPI**: Web interface

3. **Easy to Configure**
   - Model selection (mistral, llama2, etc.)
   - Chunk size and overlap
   - Number of contexts retrieved

4. **Production Ready**
   - Lazy initialization
   - Error handling
   - Logging and debugging

5. **Highly Extensible**
   - Add multiple documents
   - Conversation history
   - Authentication
   - Web search integration

---

**Happy RAG Development! 🚀**

For questions or issues, refer to this guide or check the application logs.
