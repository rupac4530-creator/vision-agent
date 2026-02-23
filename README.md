# Vision Agent 🎬🧠

![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)
![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688.svg)
![Docker](https://img.shields.io/badge/Docker-ready-2496ED.svg)

**Real-time multimodal AI video agent** that watches, listens, and understands video — built for the [WeMakeDevs Vision Possible Hackathon](https://wemakedevs.org).

> Upload or stream video → Extract frames + Transcribe audio + Detect objects → Generate AI-powered study notes, formulas, and viva questions — all in real-time.

---

## ✨ Features

| Feature | Detail |
|---|---|
| 📤 **Video Upload** | Drag-and-drop, supports MP4/MOV/WebM |
| 🖼️ **Frame Extraction** | 1 fps sampling via OpenCV |
| 🎙️ **Audio Transcription** | OpenAI Whisper API (cloud) |
| 🔍 **Object Detection** | YOLOv8 per-frame labels with confidence |
| 🧠 **AI Notes** | LLM-generated summary, concepts, formulas, viva questions |
| 💬 **Ask a Question** | Contextual QA chat over notes + transcript |
| 🧪 **Quiz Generator** | MCQs + short-answer questions with auto-scoring |
| 📐 **LaTeX Formulas** | MathJax-rendered formulas extracted from lectures |
| 📡 **Live Streaming** | Webcam streaming in 2s chunks with real-time agent responses |
| 🖼️ **Timeline Thumbnails** | Clickable frame timeline for video navigation |
| ⌨️ **Keyboard Shortcuts** | Space play/pause, arrow keys ±5s |
| ⚡ **LLM Caching** | In-memory cache for fast repeated queries |

## 🏗️ Architecture

```
┌─────────────┐    ┌──────────────────────────────────────────────┐
│  Browser UI  │───▶│              FastAPI Server                  │
│              │    │                                              │
│  Upload tab  │    │  /upload ──▶ frame_extractor.py              │
│  Demo page   │    │  /analyze ──▶ ffmpeg audio ──▶ whisper       │
│  Stream tab  │    │              ──▶ YOLOv8 detection            │
│              │    │  /generate_notes ──▶ OpenAI LLM              │
│  QA Chat     │    │  /ask ──▶ contextual QA (cached)             │
│  Quiz Modal  │    │  /generate_quiz ──▶ MCQ + short answer       │
│              │    │  /stream_chunk ──▶ instant per-chunk agent    │
│              │    │  /stream_finalize ──▶ stitch + full analyze   │
└─────────────┘    └──────────────────────────────────────────────┘
```

## 🚀 Quick Start

### Option A: Docker (recommended — no ffmpeg install needed)

```bash
# Set your API key and run
OPENAI_API_KEY="sk-..." docker compose up --build
# Open http://localhost:8000
```

### Option B: Local Setup

#### Prerequisites

- **Python 3.10+**
- **ffmpeg** installed and on PATH ([download](https://ffmpeg.org/download.html))
- **OpenAI API key** (for transcription, notes, QA, and quiz)

#### Windows PowerShell

```powershell
cd vision-agent\backend
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
$env:OPENAI_API_KEY = "sk-..."
uvicorn main:app --reload --port 8000
```

#### Linux / macOS

```bash
cd vision-agent/backend
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
export OPENAI_API_KEY="sk-..."
uvicorn main:app --reload --port 8000
```

Open **http://localhost:8000** — Upload & Analyze UI
Open **http://localhost:8000/demo** — Interactive Demo (QA, Quiz, Timeline)

> **💡 No API key?** The server runs without `OPENAI_API_KEY` — transcription returns a placeholder and notes serve pre-generated samples from `analysis/sample/`. Judges can browse the demo UI and sample outputs immediately.

## 📡 API Endpoints

| Method | Path | Description |
|---|---|---|
| GET | `/` | Upload & stream UI |
| GET | `/demo` | Interactive demo (QA, quiz, timeline) |
| POST | `/upload` | Upload video → extract frames |
| POST | `/analyze` | Full pipeline: frames + transcript + detection |
| POST | `/generate_notes?video_stem=video` | LLM notes from analysis |
| POST | `/ask` | Contextual QA over notes + transcript |
| POST | `/generate_quiz` | MCQ + short-answer quiz from notes |
| POST | `/stream_chunk` | Stream a 2-5s chunk for instant processing |
| POST | `/stream_finalize` | Stitch chunks + run full analysis |

## ⚙️ Environment Variables

| Variable | Default | Description |
|---|---|---|
| `OPENAI_API_KEY` | — | **Required** for transcription, notes, QA, quiz |
| `WHISPER_MODEL` | `tiny` | Whisper model size |
| `YOLO_MODEL` | `yolov8n.pt` | YOLO model file |
| `LLM_MODEL` | `gpt-4o-mini` | OpenAI chat model |

## 🛠️ Tech Stack

- **Backend**: Python 3.11, FastAPI, Uvicorn
- **Vision**: OpenCV, YOLOv8 (ultralytics)
- **Audio**: OpenAI Whisper API
- **LLM**: OpenAI GPT-4o-mini
- **Math**: MathJax 3 (LaTeX rendering)
- **Streaming**: ffmpeg, MediaRecorder API
- **Frontend**: Vanilla HTML/CSS/JS — dark glassmorphism
- **Deploy**: Docker, GitHub Actions CI

## 📁 Project Structure

```
vision-agent/
├── backend/
│   ├── main.py              # FastAPI app (all routes)
│   ├── frame_extractor.py   # OpenCV frame extraction
│   ├── transcribe.py        # OpenAI Whisper API transcription
│   ├── detect.py            # YOLOv8 detection
│   ├── llm_helpers.py       # LLM call wrapper with retry
│   ├── generate_notes.py    # Notes generator (with fallback)
│   ├── streaming.py         # Real-time chunk streaming
│   ├── requirements.txt     # Python dependencies
│   ├── final_test_report.txt
│   ├── .gitignore
│   ├── analysis/sample/     # Pre-generated sample outputs
│   │   ├── analysis.json
│   │   ├── notes.json
│   │   └── quiz.json
│   └── static/
│       ├── index.html       # Upload & live-stream UI
│       └── demo.html        # Interactive demo (QA, quiz, timeline)
├── .github/workflows/ci.yml # GitHub Actions CI
├── Dockerfile               # Docker build
├── docker-compose.yml       # One-command start
├── README.md
├── LICENSE                  # MIT
├── PRIVACY.md               # Data handling note
├── BLOG_POST.md             # Blog draft
├── SUBMISSION_NOTES.md      # Metrics & pitch
├── SUBMISSION_READY.txt     # Hackathon form fields
└── RELEASE_NOTES.md         # GitHub Release notes
```

## 📊 Performance Metrics

| Step | Time |
|---|---|
| Frame extraction (30s video) | ~1-2s |
| Whisper transcription (cloud) | ~2-5s |
| YOLOv8 detection (30 frames) | ~3-6s |
| LLM notes generation | ~3-8s |
| **Total pipeline** | **~10-20s** |

## License

MIT — see [LICENSE](LICENSE)

---

Built with ❤️ for the **WeMakeDevs Vision Possible Hackathon** — powered by [Vision Agents by Stream](https://getstream.io/video/vision-agents/) & [OpenAI](https://openai.com)

