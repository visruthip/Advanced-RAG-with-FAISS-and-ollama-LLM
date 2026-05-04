"""
FastAPI Backend for RAG Chat Application

This module provides the REST API for the RAG (Retrieval Augmented Generation) system.
It implements lazy initialization to avoid blocking on startup while loading large PDFs.

Features:
- Lazy pipeline initialization: PDF loads on first request, not on startup
- Thread-safe: Uses locks to ensure single initialization
- RESTful API: JSON endpoints for health check and question answering
- Error handling: Graceful error responses

Architecture:
    FastAPI Server
        ↓
    Query Handler
        ↓ (first request only)
    Initialize Pipeline
        ├─ Load PDF
        ├─ Chunk text
        ├─ Create FAISS index
        └─ Initialize Ollama LLM
        ↓
    Answer Question
        ├─ Retrieve contexts
        └─ Generate answer

Example Usage:
    POST /ask
    {
        "question": "What is ISO 27001?"
    }
    
    Response:
    {
        "answer": "ISO 27001 is an information security standard...",
        "contexts": ["ISO 27001 is...", "It covers..."]
    }
"""

from fastapi import FastAPI
from pydantic import BaseModel
import threading
import requests

from rag.data_loader import PDFReader
from rag.text_utils import text2chunk
from rag.vector_store import FAISSRetrieval
from rag.llm import OllamaLLM
from rag.pipeline import SimpleRAGPipeline
from rag.config import *

# Initialize FastAPI application
app = FastAPI(
    title="RAG Chat API",
    description="Retrieval Augmented Generation API for document Q&A",
    version="1.0.0"
)

# Global pipeline state
pipeline = None
pipeline_lock = threading.Lock()  # Ensures thread-safe initialization
pipeline_initialized = False
initialization_in_progress = False
initialization_status = "Not started"


class Query(BaseModel):
    """Request model for asking questions"""
    question: str  # User's question about the document


def initialize_pipeline():
    """
    Initialize the RAG pipeline on first use.
    
    This function loads the PDF, chunks text, creates FAISS index,
    and initializes the Ollama LLM. It's called lazily (on first request)
    rather than on startup to avoid blocking the server.
    
    Thread Safety:
    - Uses double-check locking pattern
    - Only one thread initializes the pipeline
    - Other threads wait for initialization to complete
    
    Process:
    1. Load PDF from config path
    2. Extract text from all pages
    3. Split into overlapping chunks
    4. Create FAISS vector database
    5. Initialize Ollama LLM connection
    6. Create RAG pipeline orchestrator
    """
    global pipeline, pipeline_initialized, initialization_in_progress, initialization_status
    
    # Quick check (without lock for performance)
    if pipeline_initialized:
        return
    
    # Acquire lock for initialization
    with pipeline_lock:
        # Double-check after acquiring lock
        if pipeline_initialized:
            return
        
        initialization_in_progress = True
        initialization_status = "Starting initialization..."
            
        try:
            initialization_status = "Loading PDF..."
            print("[INFO] Loading PDF from example/iso27001.pdf...")

            # 1. Load PDF
            reader = PDFReader("example/iso27001.pdf")
            texts = reader.read()
            print(f"[INFO] PDF loaded: {len(texts)} pages")

            # 2. Chunk text with overlap
            initialization_status = "Chunking text..."
            chunks = []
            for t in texts:
                chunks.extend(text2chunk(t, CHUNK_SIZE, CHUNK_OVERLAP))

            print(f"[INFO] Created {len(chunks)} chunks (CHUNK_SIZE={CHUNK_SIZE})")

            # 3. Create FAISS vector database
            initialization_status = "Building FAISS index..."
            print("[INFO] Building FAISS vector database...")
            retrieval = FAISSRetrieval(chunks)
            print("[INFO] FAISS index created successfully")

            # 4. Initialize Ollama LLM
            initialization_status = "Connecting to Ollama..."
            print(f"[INFO] Connecting to Ollama at {OLLAMA_BASE_URL} with model '{MODEL_NAME}'...")
            llm = OllamaLLM(
                model_name=MODEL_NAME,
                base_url=OLLAMA_BASE_URL
            )
            print("[INFO] Testing Ollama connection...")
            # Test connection with a simple prompt
            test_response = llm.generate("Hello")
            print("[INFO] Ollama connection successful")

            # 5. Create RAG pipeline
            initialization_status = "Creating pipeline..."
            pipeline = SimpleRAGPipeline(
                retrieval=retrieval,
                llm=llm,
                top_k=TOP_K
            )

            print("[SUCCESS] RAG Pipeline initialized and ready!")
            initialization_status = "Ready"
            pipeline_initialized = True
        except FileNotFoundError as e:
            initialization_status = f"Error: File not found - {e}"
            print(f"[ERROR] File not found: {e}")
            raise
        except requests.exceptions.ConnectionError as e:
            initialization_status = f"Error: Cannot connect to Ollama"
            print(f"[ERROR] Cannot connect to Ollama at {OLLAMA_BASE_URL}")
            print(f"[ERROR] Make sure Ollama is running: ollama serve")
            print(f"[ERROR] Details: {e}")
            raise
        except Exception as e:
            initialization_status = f"Error: {str(e)}"
            print(f"[ERROR] Pipeline initialization failed: {e}")
            import traceback
            traceback.print_exc()
            raise
        finally:
            initialization_in_progress = False


@app.get("/")
def home():
    """
    Health check endpoint.
    
    Returns:
        dict: Status indicator
    """
    return {"status": "ok"}


@app.get("/status")
def status():
    """
    Detailed status endpoint for debugging.
    
    Returns:
        dict: Information about system status
    """
    status_info = {
        "server": "ok",
        "pipeline_initialized": pipeline_initialized,
        "config": {
            "MODEL_NAME": MODEL_NAME,
            "OLLAMA_BASE_URL": OLLAMA_BASE_URL,
            "CHUNK_SIZE": CHUNK_SIZE,
            "TOP_K": TOP_K
        }
    }
    
    # Check Ollama connection
    try:
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get("models", [])
            model_names = [m["name"] for m in models]
            status_info["ollama"] = {
                "status": "ok",
                "available_models": model_names,
                "required_model": MODEL_NAME,
                "model_available": any(MODEL_NAME in m for m in model_names)
            }
        else:
            status_info["ollama"] = {"status": "error", "error": f"HTTP {response.status_code}"}
    except requests.exceptions.ConnectionError:
        status_info["ollama"] = {
            "status": "error",
            "error": f"Cannot connect to Ollama at {OLLAMA_BASE_URL}",
            "suggestion": "Run: ollama serve"
        }
    except Exception as e:
        status_info["ollama"] = {"status": "error", "error": str(e)}
    
    return status_info


@app.get("/progress")
def progress():
    """
    Check initialization progress.
    
    Returns:
        dict: Progress information
    """
    return {
        "initialized": pipeline_initialized,
        "in_progress": initialization_in_progress,
        "status": initialization_status
    }


@app.post("/ask")
def ask(query: Query):
    """
    Main endpoint for answering questions using RAG.
    
    This endpoint:
    1. Initializes pipeline on first call (if not already done)
    2. Retrieves relevant document chunks using FAISS
    3. Generates answer using Ollama LLM
    4. Returns answer with source contexts
    
    Args:
        query (Query): User question object containing 'question' field
    
    Returns:
        dict: Contains 'answer' (generated response) and 'contexts' (source chunks)
        
    Example:
        Request:
            {
                "question": "What is ISO 27001?"
            }
        
        Response:
            {
                "answer": "ISO 27001 is an international standard for information security management systems...",
                "contexts": [
                    "ISO 27001 defines requirements for an information security management system (ISMS)",
                    "It provides a framework for implementing security controls..."
                ]
            }
    
    Error Handling:
        - Returns {"error": "not ready"} if pipeline initialization fails
    """
    print(f"\n[FASTAPI] Received question: {query.question}", flush=True)
    
    # Initialize pipeline on first request
    try:
        print("[FASTAPI] Initializing pipeline...", flush=True)
        initialize_pipeline()
        print("[FASTAPI] Pipeline initialization complete", flush=True)
    except FileNotFoundError as e:
        print(f"[FASTAPI] ERROR: PDF not found: {e}", flush=True)
        return {"error": "PDF file not found at example/iso27001.pdf"}
    except requests.exceptions.ConnectionError as e:
        print(f"[FASTAPI] ERROR: Cannot connect to Ollama: {e}", flush=True)
        return {
            "error": f"Cannot connect to Ollama at {OLLAMA_BASE_URL}. Make sure Ollama is running with: ollama serve"
        }
    except Exception as e:
        print(f"[FASTAPI] ERROR: Pipeline init failed: {e}", flush=True)
        import traceback
        traceback.print_exc()
        return {"error": f"Pipeline initialization failed: {str(e)}"}
    
    # Check if pipeline is ready
    if pipeline is None:
        print("[FASTAPI] ERROR: Pipeline is None", flush=True)
        return {"error": "Pipeline failed to initialize. Check logs above."}

    try:
        print(f"[FASTAPI] Running RAG pipeline with question: {query.question}", flush=True)
        # Run RAG pipeline: retrieve contexts and generate answer
        result = pipeline.run(query.question)
        
        print(f"[FASTAPI] Pipeline result - answer_length: {len(str(result.answer))}, contexts: {len(result.contexts)}", flush=True)
        print(f"[FASTAPI] Answer preview: {str(result.answer)[:100]}", flush=True)
        
        if not result.answer or not result.answer.strip():
            print("[FASTAPI] ERROR: Empty answer generated by pipeline!", flush=True)
            return {"error": "Model generated empty response. Check Ollama logs."}

        # Return answer with source contexts
        print("[FASTAPI] SUCCESS: Returning answer", flush=True)
        return {
            "answer": result.answer,
            "contexts": result.contexts
        }
    except Exception as e:
        print(f"[FASTAPI] ERROR: Exception during generation: {e}", flush=True)
        import traceback
        traceback.print_exc()
        return {"error": f"Error generating answer: {str(e)}"}


@app.post("/ask/sample")
def ask_sample(query: Query):
    """
    Sample endpoint for testing without full initialization.
    Returns pre-generated answers for common questions.
    
    This is useful for testing the UI and application structure
    while the full RAG pipeline is initializing.
    
    Args:
        query (Query): User question object containing 'question' field
    
    Returns:
        dict: Sample answer and contexts
    """
    question = query.question.lower().strip()
    
    # Sample responses for common questions
    samples = {
        "iso 27001": {
            "answer": "ISO 27001 is an international standard for information security management systems (ISMS). It provides a systematic approach to managing sensitive company information so that it remains secure. The standard includes requirements for establishing, implementing, maintaining and continually improving an information security management system within the context of the organization.",
            "contexts": [
                "ISO/IEC 27001 specifies requirements for an information security management system (ISMS). It is designed to ensure the confidentiality, integrity and availability of information.",
                "ISO 27001 is based on a risk management approach and requires organizations to identify information security risks and implement appropriate controls to manage those risks.",
                "The standard covers all types of organizations and defines requirements for information security policies, procedures, training, and technical controls."
            ]
        },
        "information security": {
            "answer": "Information security refers to the practice of protecting information from unauthorized access, use, disclosure, disruption, modification, or destruction. It encompasses the protection of digital and physical information, as well as the systems and infrastructure used to process, transmit, and store that information.",
            "contexts": [
                "Information security is critical for organizations to protect their valuable assets, including customer data, trade secrets, and proprietary information.",
                "Security measures include physical controls (locks, surveillance), technical controls (encryption, firewalls), and administrative controls (policies, training).",
                "Information security requires a comprehensive approach involving people, processes, and technology to effectively protect organizational information."
            ]
        },
        "default": {
            "answer": "This is a sample response from the RAG system. The full AI model is currently initializing and will provide more detailed, contextual answers based on the ISO 27001 document. Please wait a moment and try again, or ask a different question.",
            "contexts": [
                "Sample context 1: The system is loading the document knowledge base...",
                "Sample context 2: Actual answers will be generated from the PDF document once initialization is complete.",
                "Sample context 3: For now, this sample response demonstrates the system's interface and response format."
            ]
        }
    }
    
    # Find matching sample response
    for key, response in samples.items():
        if key in question:
            return response
    
    # Return default sample
    return samples["default"]