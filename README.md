# Adverse Media Pipeline

This is an async FastAPI service for adverse-media screening.
It uses zero-token NLI decision stages to reduce noise before calling a generative LLM.
The LLM is used selectively for non-rejected hits and edge cases, and its output is validated.
The design and trade-offs are described in [ARCHITECTURE.md](ARCHITECTURE.md).
Recorded evaluation outputs are kept in [`docs/evaluation/`](docs/evaluation/).

## Flow

```text
sources -> dedup -> entity gate -> NLI gate -> victim flag -> tags
  -> LLM (hits + edge cases only) -> validated result
```

## Setup

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
```

### Linux/macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

The requirements file installs CPU PyTorch from its extra index line. Set
`LLM_API_KEY` in `.env`. The NLI model is downloaded from Hugging Face on first start.

## Run the server

```powershell
.\.venv\Scripts\python.exe -m uvicorn adverse_media.api:app --port 8000
```

```bash
curl -X POST http://127.0.0.1:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{"company_name":"Acme GmbH","keywords":["corruption","fraud","bribery","money laundering"],"language":"en"}'
```

```powershell
$body = @{ company_name = 'Acme GmbH'; keywords = @('corruption', 'fraud', 'bribery', 'money laundering'); language = 'en' } | ConvertTo-Json
Invoke-RestMethod -Uri 'http://127.0.0.1:8000/analyze' -Method Post -ContentType 'application/json' -Body $body | ConvertTo-Json -Depth 6
```

## Demo

```powershell
.\.venv\Scripts\python.exe scripts\demo.py
```

The demo loads the real NLI model and sends the sample request in-process.
On the Groq free tier, the rate limit can make the job take about a minute.

## Tests

```powershell
.\.venv\Scripts\python.exe -m pytest
```

The default suite uses fakes and does not load a model or call an API.
The classifier integration test requires `RUN_MODEL_TESTS=1` and downloads the model.

## Configuration

| Setting | Default | Meaning |
|---|---:|---|
| `GATE_REJECT_THRESHOLD` | `0.50` | Reject below this NLI relevancy score. |
| `VICTIM_VETO_THRESHOLD` | `0.30` | Flag possible victim or cleared-party articles. |
| `TAG_THRESHOLD` | `0.50` | Minimum keyword-tag probability. |
| `TAG_MAX_LABELS` | `2` | Maximum returned NLI categories. |
| `NLI_MODEL_NAME` | `MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7` | Hugging Face NLI model. |
| `LLM_BASE_URL` | `https://api.groq.com/openai/v1` | OpenAI-compatible endpoint. |
| `LLM_API_KEY` | empty | Provider key; empty uses the fallback. |
| `LLM_MODEL` | `llama-3.1-8b-instant` | Provider model name. |
| `LLM_MAX_CONCURRENCY` | `4` | Maximum concurrent LLM calls. |
| `LLM_TIMEOUT_SECONDS` | `20.0` | Per-request timeout. |
| `LLM_JSON_MODE` | `true` | Request JSON-object responses. |
| `LLM_MAX_TOKENS` | `1024` | Maximum completion tokens. |
| `LLM_REASONING_EFFORT` | unset | Optional reasoning-model setting. |
| `SOURCE_TIMEOUT_SECONDS` | `5` | Per-source fetch timeout. |
| `DATA_PATH` | `data/articles.json` | Local candidate dataset path. |

The `.env.example` demo profile sets concurrency to `1` for the free-tier rate limit.
The application default in `config.py` remains `4`.

## Swapping the LLM backend

Any OpenAI-compatible endpoint can be selected with `LLM_BASE_URL`, `LLM_MODEL`,
and `LLM_API_KEY`, including Groq, Gemini's OpenAI-compatible endpoint, Ollama,
or vLLM. Ollama and vLLM are supported by configuration only and were not benchmarked.

## Known limitations

- The evaluation set is small: 40 articles, with DEV used for tuning.
- The free-tier provider rate limit can dominate latency for larger jobs.
- If the LLM is unavailable, the fallback can return unverified articles marked `NOT verified by an LLM`.
- The entity gate cannot distinguish `Acme Corp` from `Acme GmbH`; the LLM handles that distinction.
- NLI inference is CPU-bound and the first model load is slow.

See [ARCHITECTURE.md](ARCHITECTURE.md) for the detailed design and limitations.
