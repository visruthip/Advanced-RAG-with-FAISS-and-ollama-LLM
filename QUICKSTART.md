# ⚡ Quick Start Guide

## One-Command Startup (Easiest)

Open PowerShell and run:

```powershell
cd C:\Users\VISRUTHI\Documents\GenAI\RAG_FAISS_LLAMA
.\rfl\Scripts\Activate.ps1
python run.py
```

Wait ~10 seconds, then open: **http://localhost:5000**

---

## Manual Startup (Two Terminals)

**Terminal 1:**
```powershell
cd C:\Users\VISRUTHI\Documents\GenAI\RAG_FAISS_LLAMA
.\rfl\Scripts\Activate.ps1
python -m uvicorn app:app --host 127.0.0.1 --port 8000 --reload
```

**Terminal 2:**
```powershell
cd C:\Users\VISRUTHI\Documents\GenAI\RAG_FAISS_LLAMA
.\rfl\Scripts\Activate.ps1
python app_flask.py
```

Then open: **http://localhost:5000**

---

## Prerequisites Checklist

- ✅ Python 3.11+ installed
- ✅ Ollama running (`ollama serve` or Windows service)
- ✅ Ollama model downloaded (`ollama pull llama2`)

---

## Stop the Application

Press **Ctrl+C** in the terminal(s)

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| Cannot connect to backend | Make sure FastAPI (Terminal 1) is running |
| Ollama connection error | Start Ollama: `ollama serve` |
| Port already in use | `taskkill /F /IM python.exe` then restart |
| Module not found | `pip install -r requirements.txt` |

---

## Next Steps

- Read [README.md](README.md) for detailed architecture
- Read [SETUP.md](SETUP.md) for comprehensive setup guide
- Edit [rag/config.py](rag/config.py) to customize behavior
