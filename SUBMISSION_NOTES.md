# Vision Agent — Hackathon Submission

## One-line pitch

Real-Time Lecture Agent — an end-to-end multimodal system that ingests recorded video, extracts frames & audio, does low-latency chunked processing, and uses LLMs to produce concise notes, LaTeX formulas, and viva-style questions with provenance.

## Important files

| File | Purpose |
|---|---|
| `backend/main.py` | FastAPI server with all endpoints |
| `backend/static/demo.html` | Interactive demo UI |
| `backend/static/index.html` | Upload & live-streaming UI |
| `README.md` | Setup, architecture, and usage |

## Metric highlights

| Metric | Value |
|---|---|
| ASR model | whisper-1 (cloud) |
| Vision model | yolov8n |
| LLM model | gpt-4o-mini |
| Per-chunk latency | ~2-5s (laptop) |
| Demo video length | 2:30 |

## How to reproduce

```bash
# 1. Install ffmpeg, Python 3.10+
# 2. Create venv and install dependencies
cd vision-agent/backend
python -m venv venv
source venv/bin/activate  # Windows: .\venv\Scripts\Activate.ps1
pip install -r requirements.txt

# 3. Set API key
export OPENAI_API_KEY="sk-..."

# 4. Run
uvicorn main:app --reload --port 8000

# 5. Open http://localhost:8000
```

## Thank you

Thank you to the WeMakeDevs Vision Possible hackathon organizers and the Vision Agents SDK by Stream for the tools and inspiration.
