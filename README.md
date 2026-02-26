# Vision Agent 🎬🧠

![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)
![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688.svg)
![Docker](https://img.shields.io/badge/Docker-ready-2496ED.svg)
![YOLO](https://img.shields.io/badge/YOLOv8-detection-orange.svg)
![Gemini](https://img.shields.io/badge/Gemini-AI-4285F4.svg)

**Real-time multimodal AI video agent** — watches, listens, detects, and reasons about video in real-time with bounding box overlays and 2-tier agent intelligence.

> Stream live video → YOLOv8 detects objects with bounding boxes → Agent gives instant deterministic reply (<500ms) + polished LLM reply with provenance → All measured and displayed in real-time.

Built for the [WeMakeDevs Vision Possible Hackathon](https://wemakedevs.org) — powered by [Vision Agents SDK](https://getstream.io/video/vision-agents/).

---

## ✨ Features

| Feature | Detail |
|---|---|
| 📡 **Live Webcam Stream** | 2s chunk streaming with real-time YOLO detection + bounding box canvas overlays |
| 🎯 **Bounding Box Overlays** | Canvas-drawn bboxes with labels, confidence %, and per-label color coding |
| 📊 **Real-Time Metrics** | 6-metric dashboard: chunks, avg latency, P90, model FPS, frames, objects |
| 🤖 **2-Tier Agent** | FastReply (<500ms, deterministic) + PolishReply (LLM, ~3-8s, background) |
| 🔗 **Provenance Links** | Every agent response cites its sources: detection data, transcript, notes |
| 📤 **Video Upload** | Drag-and-drop MP4/MOV/WebM, full pipeline analysis |
| 🎙️ **Audio Transcription** | OpenAI Whisper API (cloud) |
| 🔍 **Object Detection** | YOLOv8n per-frame with full `[x1,y1,x2,y2]` bounding boxes |
| 🧠 **AI Notes** | LLM-generated summary, concepts, formulas, viva questions |
| 💬 **Click-to-Ask** | Click a detected label → agent answers with context |
| 🧪 **Quiz Generator** | MCQs + short-answer questions with auto-scoring |
| ⚡ **Dual LLM** | Gemini + OpenAI with auto-fallback + quota handling |
| 🌐 **URL Ingestion** | Paste YouTube/Vimeo URL → auto-download + full analysis |
| 📐 **LaTeX Formulas** | MathJax-rendered formulas from lectures |

## 🏗️ Architecture

```
┌──────────────┐    ┌───────────────────────────────────────────────────┐
│  Browser UI  │───▶│                FastAPI Server                     │
│              │    │                                                   │
│  🎥 Live     │    │  /stream_chunk ──▶ vision_worker.py (YOLO)       │
│  Stream +    │    │                    ↳ bboxes + latency tracking    │
│  Canvas      │    │  /stream_status ──▶ real-time metrics            │
│  Overlays    │    │                                                   │
│              │    │  /ask ──▶ agent_core.py                           │
│  📊 Metrics  │    │           ├─ Tier A: FastReply (<500ms)           │
│  Dashboard   │    │           └─ Tier B: PolishReply (LLM + jobs.py) │
│              │    │                                                   │
│  🤖 Agent    │    │  /analyze ──▶ ffmpeg ──▶ whisper ──▶ YOLO        │
│  Chat Panel  │    │  /generate_notes ──▶ llm_provider.py (async job) │
│              │    │  /ingest_url ──▶ yt-dlp ──▶ full pipeline        │
└──────────────┘    └───────────────────────────────────────────────────┘
```

### 2-Tier Agent Intelligence

```
User clicks "person" label
         │
         ▼
┌─ Tier A: FastReply ──────────────┐
│ Template + YOLO labels + transcript │  < 500ms
│ "I see person ×3 in the video"   │  deterministic
│ Source: detection, transcript    │  cached
└──────────────────────────────────┘
         │
         ▼ (background LLM job)
┌─ Tier B: PolishReply ────────────┐
│ Full context → Gemini/OpenAI     │  ~3-8s
│ Detailed analysis with reasoning │  provenance links
│ Auto-fallback on quota/timeout   │  polls /jobs/{id}
└──────────────────────────────────┘
```

## 🚀 Quick Start

### Option A: Docker (recommended)

```bash
OPENAI_API_KEY="sk-..." GEMINI_API_KEY="..." docker compose up --build
# Open http://localhost:8000
```

### Option B: Local Setup

#### Prerequisites
- **Python 3.10+**
- **ffmpeg** on PATH ([download](https://ffmpeg.org/download.html))
- **API key**: OpenAI and/or Gemini

#### Windows PowerShell

```powershell
cd vision-agent\backend
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
$env:OPENAI_API_KEY = "sk-..."
$env:GEMINI_API_KEY = "..."
uvicorn main:app --reload --port 8000
```

#### Linux / macOS

```bash
cd vision-agent/backend
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
export OPENAI_API_KEY="sk-..." GEMINI_API_KEY="..."
uvicorn main:app --reload --port 8000
```

> **💡 No API key?** Server runs fine — transcription returns placeholder, notes use pre-generated samples. Judges can see the full UI and stream demo immediately.

## 📡 API Endpoints

| Method | Path | Description |
|---|---|---|
| GET | `/` | Upload & live-stream UI |
| GET | `/demo` | Interactive demo (QA, quiz, timeline) |
| POST | `/upload` | Upload video → extract frames |
| POST | `/analyze` | Full pipeline: frames + transcript + detection |
| POST | `/generate_notes` | Async LLM notes generation (returns job_id) |
| POST | `/ask` | **2-tier agent**: fast reply + background LLM polish |
| POST | `/generate_quiz` | MCQ + short-answer quiz |
| POST | `/stream_chunk` | Stream 2s chunk → YOLO bboxes + transcript |
| GET | `/stream_status` | Real-time metrics: latency, FPS, detections |
| POST | `/stream_finalize` | Stitch chunks + full analysis |
| POST | `/ingest_url` | Download from URL + full pipeline |
| GET | `/jobs/{id}` | Poll async job progress |

## ⚙️ Environment Variables

| Variable | Default | Description |
|---|---|---|
| `OPENAI_API_KEY` | — | Transcription (Whisper), notes, QA, quiz |
| `GEMINI_API_KEY` | — | Gemini LLM for notes + agent reasoning |
| `LLM_PROVIDER` | `auto` | `gemini`, `openai`, or `auto` (try both) |
| `CLOUDFLARE_ACCOUNT_ID` | — | Cloudflare Workers AI account id (optional) |
| `CLOUDFLARE_API_TOKEN` | — | Cloudflare Workers AI API token (optional) |
| `CLOUDFLARE_MODEL` | `@cf/qwen/qwen1.5-14b-chat-awq` | Cloudflare Workers AI model id |
| `GEMINI_MODEL` | `gemini-2.0-flash` | Gemini model |
| `OPENAI_MODEL` | `gpt-4o-mini` | OpenAI chat model |
| `YOLO_MODEL` | `yolov8n.pt` | YOLO model file (nano/small/medium) |

### Live Coach (Pose + Rep Counting)

The Live Stream tab now includes a **fitness coach panel** (exercise selection, rep counter, form cues, and optional browser voice feedback).

- **Backend dependency**: `mediapipe` (installed via `backend/requirements.txt`). If it is missing, the rest of the app still works, but rep counting will be disabled.
- **No paid TTS required**: the UI uses the browser's built-in Speech Synthesis when enabled.

## 🛠️ Tech Stack

- **Backend**: Python 3.11, FastAPI, Uvicorn
- **Vision**: YOLOv8 (ultralytics), OpenCV, `vision_worker.py` singleton
- **Audio**: OpenAI Whisper API
- **LLM**: Gemini + OpenAI with provider abstraction (`llm_provider.py`)
- **Agent**: 2-tier reasoning engine (`agent_core.py`)
- **Jobs**: Thread-safe async job store (`jobs.py`)
- **Math**: MathJax 3 (LaTeX rendering)
- **Streaming**: ffmpeg, MediaRecorder API, growing-file WebM strategy
- **Frontend**: Vanilla HTML/CSS/JS, Canvas API for bounding boxes, dark glassmorphism
- **Deploy**: Docker, GitHub Actions CI

## 📁 Project Structure

```
vision-agent/
├── backend/
│   ├── main.py              # FastAPI app (all routes)
│   ├── agent_core.py        # 2-tier agent: FastReply + PolishReply
│   ├── vision_worker.py     # Singleton YOLO worker with latency tracking
│   ├── detect.py            # YOLOv8 detection with bounding boxes
│   ├── frame_extractor.py   # OpenCV frame extraction
│   ├── transcribe.py        # OpenAI Whisper transcription
│   ├── llm_provider.py      # Gemini/OpenAI provider abstraction
│   ├── jobs.py              # Thread-safe async job store
│   ├── streaming.py         # Real-time chunk streaming + /stream_status
│   ├── url_ingest.py        # URL download + pipeline
│   ├── requirements.txt
│   ├── analysis/sample/     # Pre-generated outputs for judges
│   └── static/
│       ├── index.html       # Upload & live-stream UI with canvas overlays
│       └── demo.html        # Interactive demo (QA, quiz, timeline)
├── Dockerfile
├── docker-compose.yml
├── README.md
├── SUBMISSION_NOTES.md
├── BLOG_POST.md
└── LICENSE
```

## 📊 Performance Metrics

| Step | Time | Notes |
|---|---|---|
| Frame extraction (30s video) | ~1-2s | OpenCV at 1 fps |
| Whisper transcription (cloud) | ~2-5s | Depends on audio length |
| YOLOv8 detection (30 frames) | ~3-6s | Nano model, CPU |
| LLM notes generation | ~3-8s | Gemini 2.0 Flash or GPT-4o-mini |
| FastReply (agent Tier A) | **<500ms** | Deterministic, cached |
| PolishReply (agent Tier B) | ~3-8s | Full LLM analysis |
| Stream chunk (end-to-end) | ~2-5s | Transcode + extract + detect |
| **Total pipeline** | **~10-20s** | Full upload → analysis |

## License

MIT — see [LICENSE](LICENSE)

---

Built with ❤️ for the **WeMakeDevs Vision Possible Hackathon** — powered by [Vision Agents SDK](https://getstream.io/video/vision-agents/), [Gemini](https://ai.google.dev/), & [OpenAI](https://openai.com)
