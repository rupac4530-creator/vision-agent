# Vision Agent v1.0 — Hackathon Submission

## One-line pitch

Real-Time Lecture Agent — an end-to-end multimodal system that ingests recorded video, extracts frames & audio, does low-latency chunked processing, and uses LLMs to produce concise notes, LaTeX formulas, and viva-style questions with provenance.

## Features

- 📤 Video upload with drag-and-drop
- 🎙️ Audio transcription (OpenAI Whisper API)
- 🔍 Per-frame object detection (YOLOv8n)
- 🧠 LLM-generated study notes with LaTeX formulas
- 💬 Contextual QA chat with provenance citations
- 🧪 MCQ + short-answer quiz generator
- 📡 Real-time webcam streaming with per-chunk feedback
- 🖼️ Clickable timeline thumbnails
- ⌨️ Keyboard shortcuts (←/→ ±5s, Space play/pause)

## Metrics

| Metric | Value |
|---|---|
| Per-chunk latency | ~2-5s |
| Full analysis (30s video) | ~10-20s |
| ASR model | whisper-1 (cloud) |
| Vision model | yolov8n |
| LLM model | gpt-4o-mini |

## How to run

```bash
# Docker (recommended)
docker compose up --build
# Open http://localhost:8000

# Or local
cd vision-agent/backend
pip install -r requirements.txt
export OPENAI_API_KEY="sk-..."
uvicorn main:app --reload --port 8000
```

## Sample outputs

Pre-generated sample outputs are available at `backend/analysis/sample/` for offline review:
- `analysis.json` — full pipeline output
- `notes.json` — LLM-generated study notes
- `quiz.json` — MCQ + short-answer questions

## Thank you

Built for the **WeMakeDevs Vision Possible Hackathon** — powered by Vision Agents by Stream and OpenAI.
