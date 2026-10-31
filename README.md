# Voice-Enabled RAG System

- **Live GitHub Pages App**: [https://saniya-cps.github.io/voice-rag/](https://saniya-cps.github.io/voice-rag/)
- **Repository**: [https://github.com/saniya-cps/voice-rag](https://github.com/saniya-cps/voice-rag)

An end-to-end, ultra-low latency **Voice-Enabled Retrieval-Augmented Generation (RAG) System** in Python. Users curate their own personal source database (`.pdf`, `.tsv`, `.csv`, `.txt`, `.md`, or custom research notes) and interact via spoken voice or text in English and Hindi.

## System Architecture & Technical Pillars

1. **Voice-to-Voice Pipeline**:
   - **Speech-to-Text (STT)**: High-speed transcription using `faster-whisper` with automatic language detection.
   - **Text-to-Speech (TTS)**: Natural neural speech synthesis using Microsoft `edge-tts` with language-adaptive voices (`en-US-JennyNeural` for English, `hi-IN-SwaraNeural` for Hindi).
2. **Personal Source Knowledge Base**:
   - Upload multiple documents (`.pdf`, `.tsv`, `.csv`, `.txt`, `.md`) or paste custom notes.
   - Automatic paragraph and sentence chunking with verifiable `[Source: <filename>, Chunk #<id>]` citations for every factual claim.
3. **Zero-Hallucination Guardrails**:
   - Dual semantic and keyword relevance gate. Empty notebooks or out-of-domain queries trigger an immediate refusal without speculative fabrication:
     > *"The uploaded notebook sources do not contain sufficient information to answer this question. Please upload relevant source documents to ground the answer."*
4. **Sub-200ms Retrieval Latency**:
   - In-memory **FAISS** vector search fused with **BM25Okapi** keyword search via Reciprocal Rank Fusion (RRF).
   - Real-time latency telemetry measuring STT, retrieval, LLM Time-to-First-Token (TTFT), and total end-to-end execution.
5. **Multilingual Capabilities (English & Hindi)**:
   - Uses `paraphrase-multilingual-MiniLM-L12-v2` dense embeddings, enabling cross-lingual queries (e.g., asking in Hindi and retrieving relevant passages from English documents).
6. **Responsive Dual-Theme Web Interface**:
   - Clean, professional design with zero emojis, celebratory confetti bursts, mobile haptics, and toast notifications.
   - Dynamic **Dark Mode** (Pookie Pink & Obsidian Black) and **Light Mode** (Matcha Green, Crisp White, and Slate).

---
## Tech Stack

| Architectural Layer | Technology / Library | Purpose in Project | Source File(s) |
| :--- | :--- | :--- | :--- |
| **Core Language & Runtime** | **Python 3.11+**, **NumPy**, **PyTorch** | Core backend execution, tensor math, and embedding computations | `app.py`, `src/pipeline.py` |
| **Frontend & Web UI (Local)** | **Gradio 6 (`Blocks`)**, **Custom CSS3 & Vanilla JS** | 2-column Personal Knowledge Studio UI, Dark/Light theme switcher, audio controls, and telemetry dashboard | `app.py` |
| **Frontend & Web UI (Live Cloud)** | **HTML5**, **CSS3**, **Vanilla ES6 JavaScript**, **GitHub Pages** | Live static web interface with automatic local backend bridge detection (`127.0.0.1:7860`) and in-browser fallback | `index.html`, `docs/index.html` |
| **API & Network Bridge** | **FastAPI**, **Starlette (`CORSMiddleware`)**, **Pydantic** | REST API endpoints (`/api/status`, `/api/query`, `/api/add_note`, `/api/starter`, `/api/clear`) with Private Network Access headers | `app.py` |
| **Speech-to-Text (STT)** | **`faster-whisper`** (CTranslate2 Whisper `base`), **Web Speech API** | Low-latency multilingual speech recognition (`en`, `hi`) with automatic language detection | `src/voice/stt.py`, `index.html` |
| **Text-to-Speech (TTS)** | **Microsoft `edge-tts`**, **Browser `speechSynthesis`** | Neural voice synthesis (`en-US-JennyNeural` for English, `hi-IN-SwaraNeural` for Hindi) | `src/voice/tts.py`, `index.html` |
| **Dense Vector Retrieval** | **FAISS (`faiss-cpu`)**, **Sentence-Transformers** | In-memory inner-product (`IndexFlatIP`) cosine similarity search using `paraphrase-multilingual-MiniLM-L12-v2` (384-dim) | `src/rag/vector_index.py` |
| **Lexical Keyword Retrieval** | **`rank-bm25` (`BM25Okapi`)** | Stop-word-filtered exact keyword matching for technical terms, names, and numbers | `src/rag/bm25_index.py` |
| **Hybrid Fusion & Guardrails** | **Reciprocal Rank Fusion (RRF)** | Combines Dense + BM25 rankings (`k=60`) and enforces dual-threshold anti-hallucination gating | `src/rag/hybrid_retriever.py` |
| **Document Ingestion & Chunking** | **`pypdf`**, **Python `csv` / Regex Chunking** | Parses `.pdf`, `.csv`, `.tsv`, `.txt`, and `.md` files into boundary-aware overlapping chunks (`450` chars, `75` overlap) | `src/rag/file_parser.py`, `src/rag/source_manager.py` |
| **LLM Generation & Grounding** | **Google GenAI SDK (`google-genai`) — `gemini-2.5-flash`** | Strictly grounded answer synthesis with `[Source: <filename>, Chunk #<id>]` citations and deterministic extractive fallback | `src/llm/generator.py`, `src/llm/prompts.py` |
| **Configuration & Security** | **`python-dotenv`**, **Environment Variables** | Secure `.env` API key loading, session-scoped keys, and strict `.gitignore` secret protection | `src/config.py`, `.gitignore` |
---
## Quickstart (Local)

### 1. Launch Interactive Web App
- **One-Click (Windows)**: Double-click `run.bat` in the project folder.
- **Via PowerShell**:
  ```powershell
  .\.venv\Scripts\python.exe app.py
  ```
Open `http://127.0.0.1:7860` in your browser.

### 2. Run Automated Tests & Benchmarks
```powershell
.\.venv\Scripts\python.exe test_pipeline.py
.\.venv\Scripts\python.exe benchmark.py
```
