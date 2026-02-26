# Vision Agent — Hackathon Submission Notes

## One-Line Pitch

**Real-time Vision Agent** — streams webcam video, runs YOLOv8 detection with bounding box overlays, and uses a 2-tier agent (instant FastReply + LLM PolishReply with provenance) to reason about what it sees — all with measured sub-second response times.

## Architecture Highlights

```
Browser ──WebRTC──▶ /stream_chunk ──▶ vision_worker.py (YOLO singleton)
                                      ↳ per-frame bboxes + latency
                    /stream_status ──▶ real-time metrics dashboard
                    /ask ──▶ agent_core.py
                             ├─ Tier A: FastReply (<500ms, deterministic)
                             └─ Tier B: PolishReply (Gemini/OpenAI, async)
```

## Key Files

| File | Purpose |
|---|---|
| `backend/agent_core.py` | **2-tier agent**: FastReply + PolishReply with provenance |
| `backend/vision_worker.py` | **Singleton YOLO** with latency tracking (avg/p90/FPS) |
| `backend/streaming.py` | Real-time chunk streaming + `/stream_status` metrics |
| `backend/llm_provider.py` | Gemini/OpenAI abstraction with quota handling |
| `backend/main.py` | FastAPI routes |
| `backend/static/index.html` | UI with canvas bbox overlays + 6-metric dashboard |

## Measured Performance

| Metric | Value | Notes |
|---|---|---|
| FastReply latency | **<500ms** | Deterministic, cached |
| PolishReply latency | ~3-8s | Background LLM (Gemini 2.0 Flash) |
| YOLOv8n per-frame | ~50-200ms | CPU, nano model |
| Stream chunk E2E | ~2-5s | Transcode + extract + detect |
| Vision model | YOLOv8n | 80-class COCO detection |
| LLM providers | Gemini + OpenAI | Auto-fallback on quota |

## Best Use of Vision Agents SDK

- Integrated Vision Agents SDK for real-time video pipeline
- Used `ultralytics` YOLO models via SDK plugin ecosystem
- Singleton model loading for low-latency repeated inference
- Real-time metrics dashboard (avg/p90/FPS) inspired by SDK patterns
- Canvas bounding box overlay for visual agent output

## How to Reproduce

```bash
cd vision-agent/backend
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
export OPENAI_API_KEY="sk-..." GEMINI_API_KEY="..."
uvicorn main:app --reload --port 8000
# Open http://localhost:8000 → Live Stream tab
```

## Thank You

Thank you to the WeMakeDevs Vision Possible hackathon organizers and the Vision Agents SDK by Stream for the tools and inspiration.
