# Adverse Media Pipeline

Async FastAPI service that screens news articles for adverse media about a company.
Lightweight decision models filter out noise first; the final LLM routing policy will be decided after evaluating the decision layer.

## Install

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
```

The requirements use the CPU-only PyTorch index; no CUDA installation is needed.

## Run

```powershell
uvicorn adverse_media.api:app --reload
```

Open http://127.0.0.1:8000/docs and call `POST /analyze`.

## Test

```powershell
pytest
```
