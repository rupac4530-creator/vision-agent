<p align="center">
  <img src="https://img.shields.io/badge/🏆_WeMakeDevs-Vision_Possible-blueviolet?style=for-the-badge" alt="Vision Possible"/>
  <img src="https://img.shields.io/badge/Stream-Vision_Agents-00C853?style=for-the-badge&logo=data:image/svg+xml;base64," alt="Stream"/>
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python"/>
  <img src="https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge" alt="License"/>
  <img src="https://img.shields.io/github/stars/rupac4530-creator/vision-agent?style=for-the-badge&color=gold" alt="Stars"/>
  <img src="https://img.shields.io/badge/🎓_First_Hackathon-Class_12_Student-ff69b4?style=for-the-badge" alt="First Hackathon"/>
</p>

<h1 align="center">🤖 Vision Agent — Multi-Modal AI Platform</h1>

<p align="center">
  <strong>Production-grade AI platform with 17 real-time vision & audio features, 22 SDK modules, 7-tier LLM cascade, and 37+ API endpoints.</strong>
</p>

<p align="center">
  Built for <a href="https://github.com/GetStream/Vision-Agents"><strong>Vision Possible Hackathon</strong></a> by <a href="https://www.wemakedevs.org/">WeMakeDevs</a> × <a href="https://getstream.io/">Stream</a>
</p>

<p align="center">
  <a href="#-quick-start">Quick Start</a> •
  <a href="#-features--17-ai-tabs">Features</a> •
  <a href="#-architecture">Architecture</a> •
  <a href="#-sdk-modules">SDK Modules</a> •
  <a href="#-api-reference">API</a> •
  <a href="#-contributing">Contributing</a>
</p>

---

## 📸 Platform Preview

<p align="center">
  <img src="assets/screenshots/01-upload-analyze.png" alt="Upload & Analyze" width="600"/>
  <br/><em>🎬 Upload & Analyze — Video upload, frame extraction, YOLO detection</em>
</p>

<p align="center">
  <img src="assets/screenshots/02-live-stream.png" alt="Live Stream" width="600"/>
  <br/><em>📡 Real-Time Vision Agent Stream — Webcam YOLO detection with agent reasoning</em>
</p>

<p align="center">
  <img src="assets/screenshots/03-dashboard.png" alt="Performance Dashboard" width="600"/>
  <br/><em>📊 Performance Dashboard — LLM cascade status, API latency, session stats</em>
</p>

<p align="center">
  <img src="assets/screenshots/04-ai-notes.png" alt="AI Notes" width="600"/>
  <br/><em>📝 AI Study Notes — LLM-powered structured notes from video content</em>
</p>

<p align="center">
  <img src="assets/screenshots/05-characters.png" alt="AI Character Personas" width="600"/>
  <br/><em>🧙 AI Character Personas — 6 unique AI characters with distinct personalities</em>
</p>

## 👤 About the Creator

| | Details |
|---|---|
| **Name** | **Bedanta Chatterjee** |
| **GitHub** | [@rupac4530-creator](https://github.com/rupac4530-creator) |
| **LinkedIn** | [Connect on LinkedIn](https://www.linkedin.com/in/bedanta-chatterjee-6286ba236) |
| **Country** | 🇮🇳 India |
| **School** | S.E. Rly Mixed H.S. School (Class 12) |
| **Hackathon** | [Vision Possible](https://github.com/GetStream/Vision-Agents) by WeMakeDevs × Stream |

> 🎓 **This is my first hackathon ever!** I'm a Class 12 student currently preparing for my board exams, deeply passionate about AI, coding, and building things that push the boundaries of what's possible. I believe in learning by doing — and this project is the result of that philosophy. I'm endlessly curious about multi-modal AI, real-time systems, and making technology accessible to everyone.
>
> 🛠️ **Tools & AI Assistants Used**: [Anti-Gravity](https://deepmind.google/) (Google DeepMind), [Cursor](https://cursor.sh/), and [ChatGPT](https://chat.openai.com/) — these incredible AI coding assistants helped accelerate development and made this ambitious project possible.

---

## 🚀 Quick Start

### Option A: Local (Python)

```bash
# Clone the repo
git clone https://github.com/rupac4530-creator/vision-agent.git
cd vision-agent/backend

# Create virtual environment
python -m venv venv
venv\Scripts\activate   # Windows
# source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your API keys (only GEMINI_API_KEY is required)

# Download YOLO models (auto-downloads on first run)
# Or manually: pip install ultralytics && yolo export model=yolov8n.pt

# Start the server
python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
# → Open http://localhost:8000
```

### Option B: Docker Compose

```bash
cd vision-agent
docker compose up --build -d
# → Open http://localhost:8000
```

---

## ✨ Features — 17 AI Tabs

| # | Tab | What It Does |
|---|-----|-------------|
| 1 | 🎬 **Upload & Analyze** | Video upload → frame extraction → YOLO object detection |
| 2 | 📝 **AI Notes** | LLM-powered study notes from video content |
| 3 | 💬 **Ask Anything** | Contextual Q&A over your video analysis |
| 4 | 🌐 **URL Ingest** | YouTube/web article → auto-analysis pipeline |
| 5 | 📡 **Live Stream** | Real-time streaming + AI overlay |
| 6 | 🤖 **Agent Chat** | Streaming LLM chat with vision context |
| 7 | 🧪 **Quiz** | Auto-generated MCQ quizzes from content |
| 8 | 🏋️ **Pose Coach** | Real-time YOLO pose estimation + rep counter |
| 9 | 🛡️ **Security Cam** | Object detection + threat alerts + wanted poster |
| 10 | 🧙 **Characters** | 6 AI personas (Shakespeare, Einstein, etc.) |
| 11 | 👥 **Crowd Monitor** | Crowd density + safety score + heatmap |
| 12 | 🎮 **Gaming AI** | Game screenshot → strategic advice |
| 13 | 📊 **Dashboard** | Real-time metrics + LLM cascade status |
| 14 | 🌿 **EcoWatch** | Forest/wildlife surveillance — fire & poaching detection |
| 15 | 🚗 **BlindSpot** | Dashcam AI — pedestrian/cyclist/hazard alerts |
| 16 | 📋 **Meeting AI** | Live transcription + action items + LLM summary |
| 17 | ♿ **Accessibility** | Scene descriptions + text reading + navigation + TTS |

---

## 🧠 LLM Cascade (7 Tiers)

The system automatically falls through providers for maximum reliability:

```
Tier 0  ──  GLM-5:cloud via Z.ai Ollama (local priority)
Tier 1  ──  Google Gemini 2.0 Flash (fastest free tier)  ← recommended
Tier 2  ──  GitHub Models (GPT-4o-mini)
Tier 3  ──  GitHub Models (DeepSeek-R1)
Tier 4  ──  OpenAI GPT-4o-mini
Tier 5  ──  Cloudflare Workers AI
Tier 6  ──  Groq (llama-3.3-70b-versatile)
Tier 7  ──  Auto-Summary (offline fallback)
```

---

## 🔌 SDK Modules (22 Extracted)

Full integration of the [GetStream/Vision-Agents](https://github.com/GetStream/Vision-Agents) SDK:

| Phase | Module | Description |
|-------|--------|-------------|
| 1-3 | `function_registry` | LLM tool registration with JSON schema generation |
| 1-3 | `event_bus` | SSE-enabled async event system |
| 1-3 | `observability` | Prometheus-compatible metrics + collectors |
| 1-3 | `video_processor` | Abstract processor pipeline for video frames |
| 1-3 | `conversation` | Multi-turn conversation memory |
| 4 | `llm_types` | SDK type system (ContentPart, ToolSchema, NormalizedResponse) |
| 4 | `agent_core` | Agent orchestration with tool-calling loop |
| 4 | `rag_engine` | TF-IDF RAG with document chunking |
| 4 | `instructions` | Markdown instruction loader with @-mention includes |
| 4 | `warmup_cache` | Model pre-loading with lazy deduplication |
| 4 | `mcp_tools` | MCP external tool server integration |
| 4 | `profiling` | Timing decorators with p50/p95/p99 histograms |
| 4 | `turn_detection` | Silence-based turn detection for voice |
| 4 | `transcript_buffer` | STT transcript accumulator with speaker tracking |
| 5 | `llm_base` | Abstract LLM provider with tool-calling loop |
| 5 | `stt_base` | Abstract STT with transcript events |
| 5 | `tts_base` | Abstract TTS with streaming synthesis |
| 5 | `vad` | Energy-based voice activity detection |
| 5 | `edge_types` | Portable PcmData, VideoFrame, Participant types |
| 5 | `http_transport` | WebRTC → HTTP adapter |
| 5 | `config` | Feature flags for 12 providers + 6 features |
| 5 | `event_manager` | Typed event dispatch with auto-discovery |

---

## 🏗️ Architecture

```
vision-agent/
├── backend/
│   ├── main.py                # FastAPI — 37+ endpoints
│   ├── llm_provider.py        # 7-tier LLM cascade with health tracking
│   ├── agent_core.py          # SDK agent orchestration
│   ├── function_registry.py   # LLM tool registration + MCP
│   ├── rag_engine.py          # TF-IDF RAG system
│   ├── config.py              # Feature flags + env management
│   ├── event_bus.py           # SSE event system
│   ├── observability.py       # Metrics collectors
│   ├── pose_engine.py         # YOLO pose + rep counting
│   ├── security_cam.py        # Security camera AI
│   ├── video_processor.py     # Frame processing pipeline
│   ├── llm_base.py            # Abstract LLM provider
│   ├── stt_base.py            # Abstract STT provider
│   ├── tts_base.py            # Abstract TTS provider
│   ├── vad.py                 # Voice activity detection
│   ├── http_transport.py      # WebRTC→HTTP adapter
│   ├── static/
│   │   └── index.html         # SPA with 17 AI tabs
│   ├── test_deep_sdk.py       # 50-point verification suite
│   └── requirements.txt
├── .github/
│   └── workflows/ci.yml       # Lint + test CI
├── docker-compose.yml
├── CONTRIBUTING.md
├── CODE_OF_CONDUCT.md
├── LICENSE
└── README.md
```

---

## ⚙️ Environment Variables

Only `GEMINI_API_KEY` is required. All others are optional and enable additional tiers:

```env
# Required (Tier 1)
GEMINI_API_KEY=your_gemini_key

# Optional providers
GITHUB_TOKEN=               # Tier 2-3: GitHub Models
OPENAI_API_KEY=              # Tier 4: OpenAI
CLOUDFLARE_ACCOUNT_ID=       # Tier 5: Cloudflare
CLOUDFLARE_AUTH_TOKEN=
GROQ_API_KEY=                # Tier 6: Groq
OLLAMA_URL=                  # Tier 0: Z.ai Ollama
OLLAMA_TOKEN=
OLLAMA_MODEL=glm-5:cloud

# Feature flags (SDK modules)
ENABLE_RAG=true
ENABLE_MCP=false
ENABLE_PROFILING=true
ENABLE_STT=false
ENABLE_TTS=false
```

---

## 📡 API Reference

<details>
<summary><strong>Click to expand — 37+ endpoints</strong></summary>

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | System health + LLM provider info |
| `GET` | `/metrics` | Real-time performance metrics |
| `GET` | `/modules` | All 22 loaded SDK modules |
| `GET` | `/config` | Platform configuration (safe, no secrets) |
| `POST` | `/upload` | Video upload + frame extraction |
| `POST` | `/analyze` | Full video analysis pipeline |
| `POST` | `/generate_notes` | Async LLM notes generation |
| `POST` | `/ask` | Q&A over video analysis |
| `POST` | `/generate_quiz` | MCQ + short-answer quiz |
| `POST` | `/pose_analyze` | YOLO pose + rep counting |
| `POST` | `/stt` | Speech-to-text |
| `POST` | `/character_chat` | Persona AI chat |
| `GET` | `/personas` | List available personas |
| `POST` | `/security_analyze` | Security threat detection |
| `POST` | `/wanted_poster` | Gemini Vision wanted poster |
| `POST` | `/crowd_analyze` | Crowd density + safety |
| `POST` | `/gaming_analyze` | Game screenshot analysis |
| `POST` | `/eco_analyze` | Forest/wildlife detection |
| `POST` | `/blindspot_analyze` | Dashcam hazard detection |
| `POST` | `/meeting_frame` | Video frame participant count |
| `POST` | `/meeting_transcript_add` | Add transcript + extract actions |
| `POST` | `/meeting_summarize` | LLM meeting summary |
| `POST` | `/accessibility_describe` | Scene description (5 modes) |
| `POST` | `/agent/run` | Run agent with tools |
| `POST` | `/rag/add` | Add documents to RAG |
| `POST` | `/rag/search` | Semantic search |
| `POST` | `/turn/audio_level` | Voice turn detection |
| `POST` | `/vad/process` | Voice activity detection |
| `GET` | `/rag/stats` | RAG engine stats |
| `GET` | `/profiling/stats` | Performance profiling |
| `GET` | `/warmup/stats` | Model warmup cache |
| `GET` | `/agent/stats` | Agent orchestration stats |
| `GET` | `/mcp/stats` | MCP tool server stats |
| `GET` | `/transport/stats` | HTTP transport stats |
| `GET` | `/turn/stats` | Turn detection stats |
| `GET` | `/vad/stats` | VAD stats |
| `GET` | `/transcript/stats` | Transcript buffer stats |

</details>

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| **Backend** | FastAPI + Uvicorn |
| **Vision** | YOLOv8 (Ultralytics) — object detection + pose estimation |
| **LLM** | Gemini 2.0 Flash, GPT-4o-mini, DeepSeek-R1, Groq, GLM-5 |
| **STT** | OpenAI Whisper |
| **SDK** | [GetStream/Vision-Agents](https://github.com/GetStream/Vision-Agents) — 22 extracted modules |
| **Frontend** | Vanilla HTML/CSS/JS SPA (17 tabs, zero framework) |
| **Container** | Docker + Docker Compose |
| **CI** | GitHub Actions (lint + test) |

---

## 🧪 Testing

```bash
cd backend

# Run 50-point SDK verification (all modules)
python test_deep_sdk.py

# Run full SDK test suite
python test_full_sdk.py

# Run phase-specific tests
python test_phase1.py
python test_phase2.py
python test_phase3.py
```

**Latest results**: ✅ 50/50 tests passed across all 22 SDK modules.

---

## 🤝 Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

1. Fork the repo
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 🏆 Hackathon

| | |
|---|---|
| **Event** | [Vision Possible](https://github.com/GetStream/Vision-Agents) by [WeMakeDevs](https://www.wemakedevs.org/) × [Stream](https://getstream.io/) |
| **Track** | Multi-Modal AI Agent |
| **Participant** | [Bedanta Chatterjee](https://github.com/rupac4530-creator) |
| **Upstream SDK** | [GetStream/Vision-Agents](https://github.com/GetStream/Vision-Agents) |
| **Tags** | `#VisionPossible` · `@WeMakeDevs` · `@visionagents_ai` |

---

## 🙏 Acknowledgements

- **[GetStream/Vision-Agents](https://github.com/GetStream/Vision-Agents)** — Open Vision Agents SDK (Apache-2.0). 22 modules extracted and adapted.
- **[WeMakeDevs](https://www.wemakedevs.org/)** — Hackathon organizer (founded by [Kunal Kushwaha](https://github.com/kunal-kushwaha))
- **[Stream](https://getstream.io/)** — Real-time video infrastructure sponsor
- **[Ultralytics](https://github.com/ultralytics/ultralytics)** — YOLOv8 object detection & pose estimation
- **[Google](https://ai.google.dev/)** — Gemini 2.0 Flash LLM
- **[Anti-Gravity](https://deepmind.google/)** (Google DeepMind) — AI coding assistant
- **[Cursor](https://cursor.sh/)** — AI-powered code editor
- **[ChatGPT](https://chat.openai.com/)** — Drafting guidance and planning support

---

## 📜 License

This project is licensed under the **MIT License** — see [LICENSE](LICENSE) for details.

Third-party components are attributed in [THIRD_PARTY_LICENSES.md](backend/THIRD_PARTY_LICENSES.md).

---

<p align="center">
  Made with ❤️ by <a href="https://github.com/rupac4530-creator">Bedanta Chatterjee</a> for the <a href="https://github.com/GetStream/Vision-Agents">Vision Possible</a> hackathon
  <br/>
  <a href="https://www.linkedin.com/in/bedanta-chatterjee-6286ba236">LinkedIn</a> · <a href="https://github.com/rupac4530-creator">GitHub</a> · <a href="mailto:rupac4530@gmail.com">Email</a>
</p>
