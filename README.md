# ONLYBYTES — Ambiguity Resolver

An NLP + LLM-assisted ambiguity detection, disambiguation, routing, and semantic steering system built with **FastAPI** and **React + Vite**.

---

## Key Features

### 1. Hybrid Ambiguity Detection & Resolution
- **10 Ambiguity Types + Unambiguous (`none`):** Detects `lexical`, `syntactic`, `referential`, `scope`, `semantic`, `pragmatic`, `underspecified`, `ellipsis`, `temporal_numeric`, and `cultural_domain` ambiguities using rule-based **spaCy + WordNet** detectors (`backend/detectors.py`), a **DeBERTa** classifier (`backend/classifier.py`), and LLM analysis (`backend/resolver.py`).
- **Context-Aware Self-Consistency & Conflict Detection:** Runs a standalone pass (`temperature=0`) and multi-temperature context-aware passes, flagging context conflicts when the standalone top reading and context consensus diverge beyond `CONFLICT_THRESHOLD`.
- **Calibrated 3-Way Routing (`route_record`):**
  - `AUTO_ACCEPT`: `confidence > 0.8` and `conflict == False`
  - `ACCEPT_WITH_ALTERNATIVES`: `0.5 <= confidence <= 0.8` and `conflict == False`
  - `REVIEW`: `confidence < 0.5` or `conflict == True`
- **Optional Interactive Clarification & Rewriting:** Generates optional clarification questions only for ambiguous sentences (`backend/clarification.py`), supports inline selection, custom meanings, undo/skip, and a **Review Queue**, and produces deterministic span-aware rewrites (`backend/rewrite.py`).

### 2. Multi-Format Input & Reporting
- **Input Modes:** Upload datasets (`.csv`, `.json`, `.jsonl`, `.xlsx`, `.parquet`) or evaluate single/multi-line prompts directly in the UI.
- **Exports:** Download resolved results as **JSON**, **CSV**, or **PDF** (`backend/pdf_export.py`), plus evaluation PDF reports.

### 3. Evaluation Dashboard
- **Uploaded / Prompt Evaluation:** Displays exact counts (`Ambiguous`, `Context-conflicting`, `Unambiguous`), routing breakdown, per-type metrics, sentence-level results, and a downloadable PDF report.
- **Held-Out Benchmark (`GET /evaluate`):** Evaluates on `data/test.csv` with `Detection F1`, `Precision`, `Recall`, `Resolution Accuracy`, `FPR`, `ECE`, reliability diagram, accuracy-vs-coverage curve, simulated user clarification, and a **Steering Check** (`Controllability`, `Quality Preserved`, `2D Projection Faithfulness`, and `Simulated Reviewer`).

### 4. Semantic Lab (`Semantic Lab` Tab)
- **Frozen Sentence-Transformer Steering (`backend/steering_lab.py`):** Loads frozen `sentence-transformers/all-MiniLM-L6-v2`, caches `data/train.csv` embeddings and a 2D PCA projection (`backend/lab_cache/`), and initializes from the user's uploaded/entered records via `POST /lab/init`.
- **Interactive 2D SVG Map & Live Steering (`POST /lab/steer`):** Steers embeddings toward a chosen candidate reading (`steered = normalize((1 - alpha) * e_sentence + alpha * e_target)`) with zero LLM calls, z-scored cosine similarity softmax (`T = 0.5`) fused with initial probabilities, calibrated confidence, updated routing status, training-neighbor votes, and deterministic rewrites.

---

## Project Structure

```text
ONLYBYTES/
├── backend/
│   ├── main.py              # FastAPI app & endpoints (/process, /evaluate, /clarify, /export, /lab)
│   ├── resolver.py          # LLM self-consistency, conflict detection, calibration & routing
│   ├── detectors.py         # spaCy + WordNet linguistic ambiguity detectors
│   ├── classifier.py        # DeBERTa + rule score blending
│   ├── clarification.py     # Optional inline & MCQ clarification builder
│   ├── rewrite.py           # Deterministic span-aware sentence rewriter
│   ├── evaluator.py         # Held-out evaluation suite & steering check -> results.json
│   ├── steering_lab.py      # Frozen encoder + cached train index + /lab/init & /lab/steer
│   ├── pdf_export.py        # PDF report generation (fpdf2)
│   └── requirements.txt
├── frontend/
│   ├── index.html
│   ├── vite.config.js
│   └── src/
│       ├── App.jsx
│       ├── index.css
│       └── components/
│           ├── EvaluationTab.jsx
│           └── SemanticLabTab.jsx
├── data/
│   ├── train.csv
│   └── test.csv
├── results.json
└── .env.example
```

---

## Setup & Running Locally

### 1. Environment Variables
Copy `.env.example` to `.env` in the repository root and configure your LLM provider keys (`GROQ_API_KEY`, `GEMINI_API_KEY`).

### 2. Backend (FastAPI)
```powershell
cd backend
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

### 3. Frontend (React + Vite)
```powershell
cd frontend
npm install
npm run dev -- --host 127.0.0.1 --port 5173
```

Open `http://127.0.0.1:5173` in your browser.

---

## API Endpoints

| Method | Endpoint | Description |
| --- | --- | --- |
| `GET` | `/health` | Health check |
| `POST` | `/process` | Resolve pasted text, CSV text, or string records |
| `POST` | `/process/file` | Resolve uploaded `.csv`, `.json`, `.jsonl`, `.xlsx`, `.parquet` |
| `POST` | `/clarify` | Apply user clarification choice to a record |
| `GET` | `/evaluate` | Run/retrieve held-out evaluation + steering check (`?refresh=true` to recompute) |
| `GET` | `/export/evaluation.pdf` | Download held-out evaluation PDF report |
| `POST` | `/export/resolved.pdf` | Download resolved records PDF report |
| `POST` | `/lab/init` | Initialize Semantic Lab 2D map, readings, and auto-alpha |
| `POST` | `/lab/steer` | Steer sentence embedding toward a target reading (no LLM call) |

---

## Running Tests

```powershell
cd backend
.venv\Scripts\python.exe -m pytest test_resolver.py -v
```
