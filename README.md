# 📄 RAG Chat - Retrieval Augmented Generation with ISO 27001 Document

A modern web application that allows you to ask questions about documents using advanced RAG (Retrieval Augmented Generation) with FAISS and Ollama LLM.

## Project Architecture

```
┌─────────────────────────────────────────────────────────┐
│                  Flask Web Frontend                      │
│              (HTML/CSS/JavaScript)                       │
│          Port: 127.0.0.1:5000                           │
└──────────────────────┬──────────────────────────────────┘
                       │ (HTTP Requests)
                       ▼
┌─────────────────────────────────────────────────────────┐
│            FastAPI Backend (REST API)                   │
│              Port: 127.0.0.1:8000                       │
└──────────────────────┬──────────────────────────────────┘
                       │
         ┌─────────────┴─────────────┐
         ▼                           ▼
    ┌─────────────┐          ┌──────────────┐
    │  FAISS      │          │  Ollama LLM  │
    │  Vector DB  │          │  Local Model │
    └─────────────┘          └──────────────┘
```

## 📋 Project Structure

```
RAG_FAISS_LLAMA/
├── app.py                 # FastAPI backend (REST API on :8000)
├── app_flask.py          # Flask frontend server (Web UI on :5000)
├── requirements.txt      # Python dependencies
├── pyproject.toml        # Project metadata
├── templates/
│   └── index.html        # Web UI (beautiful chat interface)
├── rag/                  # RAG Pipeline modules (production code)
│   ├── __init__.py
│   ├── config.py         # Configuration (model, chunk size, URLs)
│   ├── data_loader.py    # PDF reader & text extraction
│   ├── llm.py            # Ollama LLM integration
│   ├── pipeline.py       # RAG pipeline orchestration
│   ├── text_utils.py     # Text preprocessing & chunking
│   └── vector_store.py   # FAISS vector database
├── example/
│   └── iso27001.pdf      # Sample document for testing
├── rfl/                  # Python virtual environment
└── README.md             # This file
```

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- Ollama installed and running (for LLM inference)
- 8GB+ RAM recommended

### Step 1: Activate Virtual Environment

**On Windows (PowerShell):**
```powershell
cd C:\Users\VISRUTHI\Documents\GenAI\RAG_FAISS_LLAMA
.\rfl\Scripts\Activate.ps1
```

**On Windows (Command Prompt):**
```cmd
cd C:\Users\VISRUTHI\Documents\GenAI\RAG_FAISS_LLAMA
rfl\Scripts\activate.bat
```

**On macOS/Linux:**
```bash
cd ~/Documents/GenAI/RAG_FAISS_LLAMA
source rfl/bin/activate
```

### Step 2: Install Dependencies (First Time Only)

```bash
pip install -r requirements.txt
```

### Step 3: Configure the Application

Edit `rag/config.py` and ensure:
- `MODEL_NAME` matches your Ollama model (default: "llama2")
- `OLLAMA_BASE_URL` is correct (default: "http://127.0.0.1:11434")
- `PDF_PATH` points to your document
- Chunk size and overlap settings are appropriate

### Step 4: Start Ollama Service

In a separate terminal:
```bash
ollama serve
```

Or if Ollama is installed as a service, it may already be running.

### Step 5: Run the Application

**Terminal 1 - FastAPI Backend:**
```bash
python -m uvicorn app:app --host 127.0.0.1 --port 8000 --reload
```

Wait for the message: `Uvicorn running on http://127.0.0.1:8000`

**Terminal 2 - Flask Frontend:**
```bash
python app_flask.py
```

You should see: `Running on http://localhost:5000`

### Step 6: Access the Application

Open your browser and go to:
```
http://localhost:5000
```

You should see the chat interface. Type a question and hit "Ask" or press Enter.

## 📝 Code Explanation

### Backend (FastAPI) - `app.py`

**Key Features:**
- **Lazy Pipeline Initialization**: PDF loads on first query, not on startup (prevents blocking)
- **Thread-Safe**: Uses locks to ensure pipeline is initialized only once
- **REST API**: Provides `/ask` endpoint for question answering

**Endpoints:**
- `GET /`: Health check
- `POST /ask`: Submit a question and get an answer with contexts

**Key Functions:**
- `initialize_pipeline()`: Loads PDF, chunks text, creates FAISS index, initializes Ollama LLM
- `ask(query)`: Processes user question through RAG pipeline

### Frontend (Flask) - `app_flask.py`

**Components:**
- Serves static HTML/CSS/JavaScript interface
- Proxies requests to FastAPI backend
- Handles connection errors gracefully

**Endpoints:**
- `GET /`: Serves main chat UI
- `POST /api/ask`: Forwards questions to FastAPI backend

### Web Interface - `templates/index.html`

**Features:**
- Beautiful gradient UI with modern design
- Real-time loading spinner
- Answer display with formatting
- Relevant contexts shown below
- Error handling with user feedback
- Responsive design (works on mobile)
- Keyboard support (Enter to submit)

### RAG Pipeline - `rag/` directory

#### `config.py`
Contains all configuration constants:
- Model names and URLs
- Chunk size and overlap
- Top-K for retrieval
- Paths to documents

#### `data_loader.py`
- `PDFReader`: Extracts text from PDF files page by page
- Handles corrupted pages gracefully

#### `text_utils.py`
- `text2chunk()`: Splits text into overlapping chunks for context preservation
- Maintains chunk relationships through overlap

#### `vector_store.py`
- `FAISSRetrieval`: FAISS-based semantic search
- Stores document chunks as embeddings
- Retrieves top-K similar chunks for a query
- Uses `sentence-transformers` for embeddings

#### `llm.py`
- `OllamaLLM`: Interface to Ollama LLM
- Generates answers based on context
- Customizable prompts and parameters

#### `pipeline.py`
- `SimpleRAGPipeline`: Orchestrates the full pipeline
- Retrieves relevant contexts
- Passes to LLM for answer generation
- Returns answer with source contexts

#### `prompt.py`
- System prompts for the LLM
- Instructs model how to answer questions

## 🔧 Configuration

Edit `rag/config.py` to customize:

```python
# LLM Settings
MODEL_NAME = "llama2"  # Ollama model to use
OLLAMA_BASE_URL = "http://127.0.0.1:11434"

# Document Processing
CHUNK_SIZE = 500      # Characters per chunk
CHUNK_OVERLAP = 100   # Overlap between chunks

# Retrieval Settings
TOP_K = 5             # Number of contexts to retrieve

# Document Path
PDF_PATH = "example/iso27001.pdf"
```

## 🔄 How the RAG Pipeline Works

1. **Document Loading**: PDF is read and text is extracted
2. **Chunking**: Text is split into overlapping chunks (preserves context)
3. **Embedding**: Each chunk is converted to vector embeddings (sentence-transformers)
4. **Indexing**: Vectors are stored in FAISS (efficient similarity search)
5. **Query Processing**: User question is embedded using the same model
6. **Retrieval**: FAISS finds top-K most similar chunks to the query
7. **Answer Generation**: LLM generates answer using retrieved chunks as context
8. **Response**: Answer and source contexts are returned to user

## 📊 Performance Tips

- **Increase TOP_K** for better context but slower responses
- **Reduce CHUNK_SIZE** for more granular results but more API calls
- **Use GPU** if available: Ensure PyTorch is using CUDA for faster embeddings
- **Larger Models** in Ollama for better answers (but slower): "mistral", "neural-chat"
- **Smaller Models** for speed: "orca-mini" (3GB)

## 🐛 Troubleshooting

### "Cannot connect to backend"
- Ensure FastAPI is running on port 8000
- Check: `netstat -ano | findstr :8000`
- Restart FastAPI if needed

### "Ollama connection refused"
- Ensure Ollama service is running
- Check Ollama is on `http://127.0.0.1:11434`
- Run `ollama serve` in a terminal

### "No answer returned"
- Check if the PDF was loaded (first query takes longer)
- Verify FAISS has chunks indexed
- Check Ollama model is working: `ollama list`

### "Slow responses"
- First query loads PDF (takes time)
- Consider using faster Ollama model
- Reduce TOP_K in config

### "Port already in use"
```bash
# Find process on port 5000
netstat -ano | findstr :5000

# Kill process by PID
taskkill /PID <PID> /F
```

## 📦 Deactivate Virtual Environment

When done, deactivate the virtual environment:

```bash
deactivate
```

## 🔐 Security Notes

- This is development-grade. For production, use:
  - Gunicorn/Waitress instead of Flask debug server
  - Proper authentication/authorization
  - HTTPS/SSL certificates
  - Rate limiting
  - Input validation

## 📄 License

This project is provided as-is for educational and development purposes.

## 🤝 Support

For issues or questions, check:
1. FastAPI documentation: https://fastapi.tiangolo.com/
2. FAISS documentation: https://faiss.ai/
3. Ollama documentation: https://ollama.ai/
