"""
Flask Frontend for RAG Chat Application

This module provides the web interface and proxies requests to the FastAPI backend.

Responsibilities:
- Serves HTML/CSS/JavaScript UI
- Handles HTTP requests from the browser
- Proxies API calls to FastAPI backend
- Error handling and user feedback

Architecture:
    Browser (http://localhost:5000)
        ↓
    Flask Server
        ├─ GET  / → Serve index.html (chat UI)
        └─ POST /api/ask → Proxy to FastAPI /ask endpoint
        ↓
    FastAPI Backend (http://127.0.0.1:8000)
        ↓
    RAG Pipeline

Routes:
- GET  /         : Serves the web UI (templates/index.html)
- POST /api/ask  : Accepts question, proxies to FastAPI, returns answer

Error Handling:
- Connection errors to backend
- Invalid request format
- Server timeouts
"""

from flask import Flask, render_template, request, jsonify
import requests
import json
import sys

# Initialize Flask application
app = Flask(__name__)

# FastAPI backend endpoint
API_URL = "http://127.0.0.1:8000"


@app.route('/api/status', methods=['GET'])
def check_status():
    """
    Check the status of the backend and its dependencies.
    
    Returns:
        dict: Status information for debugging
    """
    try:
        response = requests.get(
            f"{API_URL}/status",
            timeout=5
        )
        if response.status_code == 200:
            return jsonify(response.json())
        else:
            return jsonify({'error': 'Backend error'}), response.status_code
    except requests.exceptions.ConnectionError:
        return jsonify({
            'error': 'Cannot connect to backend',
            'suggestion': 'Make sure to run: python run.py',
            'backend_url': API_URL
        }), 500
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/progress', methods=['GET'])
def check_progress():
    """
    Check initialization progress of RAG pipeline.
    
    Returns:
        dict: Progress information including status and completion state
    """
    try:
        response = requests.get(
            f"{API_URL}/progress",
            timeout=5
        )
        if response.status_code == 200:
            return jsonify(response.json())
        else:
            return jsonify({'error': 'Backend error'}), response.status_code
    except requests.exceptions.ConnectionError:
        return jsonify({
            'error': 'Cannot connect to backend'
        }), 500
    except Exception as e:
        return jsonify({'error': str(e)}), 500




def fallback_to_sample(question):
    """
    Fallback to sample responses when full RAG pipeline is not available.
    This allows users to test the UI while initialization is in progress.
    
    Args:
        question (str): User's question
    
    Returns:
        dict: JSON response with sample answer and contexts
    """
    print(f"[FLASK] Using sample endpoint for question: {question}", flush=True)
    try:
        response = requests.post(
            f"{API_URL}/ask/sample",
            json={"question": question},
            timeout=5
        )
        if response.status_code == 200:
            return jsonify({
                **response.json(),
                "is_sample": True,
                "note": "This is a sample response. Full AI responses will be available once initialization completes."
            })
    except Exception as e:
        print(f"[FLASK] Sample endpoint failed: {e}", flush=True)
    
    # Return default sample response
    return jsonify({
        "answer": "The RAG system is initializing. This is a sample response to demonstrate the interface. Please wait a moment and refresh to get real answers from the document.",
        "contexts": [
            "The system is currently loading the PDF document and building the knowledge base.",
            "Once initialization completes, you'll get contextual answers from the ISO 27001 document.",
            "Real responses will appear here based on the actual document content."
        ],
        "is_sample": True,
        "note": "Waiting for full pipeline initialization..."
    })


@app.route('/')
def index():
    """
    Serve the main chat interface.
    
    Returns:
        HTML: The chat UI template (templates/index.html)
    """
    return render_template('index.html')


@app.route('/api/ask', methods=['POST'])
def ask_question():
    """
    Handle question submission from the web interface.
    
    This endpoint:
    1. Receives JSON question from browser
    2. Validates the input
    3. Proxies request to FastAPI backend
    4. Returns answer and contexts to browser
    5. Handles errors gracefully
    
    Request Body:
        {
            "question": "What is ISO 27001?"
        }
    
    Response:
        {
            "answer": "ISO 27001 is...",
            "contexts": ["Context 1", "Context 2"]
        }
    
    Error Responses:
        400 Bad Request: Invalid or empty question
        500 Internal Server: Backend connection error or server error
    """
    try:
        print("\n[FLASK] New question received", flush=True)
        
        # Extract question from request
        data = request.json
        question = data.get('question', '')
        print(f"[FLASK] Question: {question}", flush=True)
        
        # Validate question
        if not question:
            print("[FLASK] ERROR: Question is empty", flush=True)
            return jsonify({'error': 'Question is empty'}), 400
        
        print(f"[FLASK] Calling backend at {API_URL}/ask", flush=True)
        
        # Try full RAG pipeline with a 30-second timeout to avoid long waits
        response = requests.post(
            f"{API_URL}/ask",
            json={"question": question},
            timeout=30  # Reduced from 120s to 30s
        )
        
        print(f"[FLASK] Backend response status: {response.status_code}", flush=True)
        print(f"[FLASK] Backend response text: {response.text[:500]}", flush=True)
        
        # Check response status
        if response.status_code == 200:
            backend_data = response.json()
            print(f"[FLASK] Backend returned data: answer_length={len(str(backend_data.get('answer', '')))}, contexts={len(backend_data.get('contexts', []))}", flush=True)
            
            # Check for error responses from backend
            if 'error' in backend_data:
                print("[FLASK] Backend returned error, trying sample endpoint...", flush=True)
                return fallback_to_sample(question)
            
            if not backend_data.get('answer'):
                print("[FLASK] WARNING: Backend returned empty answer!", flush=True)
                return fallback_to_sample(question)
            
            print("[FLASK] SUCCESS: Returning answer to frontend", flush=True)
            return jsonify(backend_data)
        else:
            print(f"[FLASK] Backend error: {response.status_code}", flush=True)
            return fallback_to_sample(question)
            
    except requests.exceptions.ConnectionError as e:
        print(f"[FLASK] ERROR: Cannot connect to backend: {e}", flush=True)
        # Backend is not reachable
        return jsonify({
            'error': 'Cannot connect to backend. Make sure FastAPI is running on port 8000'
        }), 500
    except requests.exceptions.Timeout as e:
        print(f"[FLASK] ERROR: Request timeout, using sample responses: {e}", flush=True)
        # Request took too long, fallback to samples
        return fallback_to_sample(question)
    except Exception as e:
        print(f"[FLASK] ERROR: Unexpected error: {e}", flush=True)
        import traceback
        traceback.print_exc()
        # Other errors
        return jsonify({'error': f'Server error: {str(e)}'}), 500


if __name__ == '__main__':
    """
    Run the Flask development server.
    
    This starts the web interface on http://localhost:5000
    Make sure FastAPI backend is running on http://127.0.0.1:8000
    """
    print("\n")
    print("╔════════════════════════════════════════════════════════════╗")
    print("║           🌐 Flask Frontend - RAG Chat App                  ║")
    print("╚════════════════════════════════════════════════════════════╝")
    print("\n📍 Web Interface: http://localhost:5000")
    print("🔌 Backend API: http://127.0.0.1:8000")
    print("\nMake sure FastAPI backend is running!")
    print("Press Ctrl+C to stop the server\n")
    
    app.run(
        host='localhost',
        port=5000,
        debug=True,
        use_reloader=True
    )
