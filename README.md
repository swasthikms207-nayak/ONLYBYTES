# ONLYBYTES — Ambiguity Resolver

Batch text ambiguity resolver with a **FastAPI** backend (`POST /process`) and a **React + Vite** frontend.

## Run Backend (FastAPI)

```powershell
cd backend
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

## Run Frontend (React + Vite)

```powershell
cd frontend
npm install
npm run dev
```

Open `http://localhost:5173`.
