import os
from dotenv import load_dotenv

load_dotenv()

# Model: Use mistral instead of llama3 for faster performance
# Options: mistral (fast), neural-chat, orca-mini (very fast), llama3 (slower but better quality)
MODEL_NAME = os.getenv("MODEL_NAME", "mistral")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")

# Performance optimized settings
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 150))      # Reduced from 200 for faster retrieval
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 30))  # Reduced from 50
TOP_K = int(os.getenv("TOP_K", 3))                  # Reduced from 5 for faster RAG