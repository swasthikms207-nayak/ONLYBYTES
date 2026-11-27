# ONLYBYTES

Full-stack starter project with a **FastAPI** backend and a **React + Vite** frontend.

## Project Structure

```text
ONLYBYTES/
├── backend/          # FastAPI API server
│   ├── main.py
│   └── requirements.txt
├── frontend/         # React + Vite web app
│   ├── index.html
│   ├── package.json
│   ├── vite.config.js
│   └── src/
├── .env              # Local environment variables (gitignored)
├── .env.example      # Example environment template
└── README.md
```

## Setup

### 1. Configure Environment Variables

Update `.env` in the project root with your LLM API key:

```env
LLM_API_KEY=your_llm_api_key_here
LLM_API_BASE_URL=https://api.openai.com/v1
LLM_MODEL=gpt-4o-mini
```

### 2. Run the Backend (FastAPI)

```powershell
cd backend
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

- API Health Check: `http://127.0.0.1:8000/api/health`
- API Docs: `http://127.0.0.1:8000/docs`

### 3. Run the Frontend (React + Vite)

In a second terminal:

```powershell
cd frontend
npm install
npm run dev
```

- Frontend URL: `http://localhost:5173`
