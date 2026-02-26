# Building a Real-Time Vision Agent with 2-Tier Intelligence

For the **WeMakeDevs Vision Possible hackathon**, I built **Vision Agent** — a real-time multimodal system that streams webcam video, runs YOLO object detection with bounding box overlays, and uses a 2-tier agent architecture to reason about what it sees: instant deterministic replies + polished LLM analysis with provenance.

## Why This Matters

Most video AI tools process after the fact. Vision Agent processes **during** — each 2-second chunk is analyzed in real-time. Click any detected object and get an instant agent response. This is what a real vision agent should feel like.

## Architecture: The 2-Tier Agent

```
User clicks "person" in live stream
         │
    ┌────▼── FastReply (Tier A) ──────┐
    │ Template + YOLO + transcript    │ < 500ms
    │ "I see person ×3, context..."   │ deterministic
    └─────────────────────────────────┘
         │
    ┌────▼── PolishReply (Tier B) ────┐
    │ Full context → Gemini/OpenAI    │ ~3-8s
    │ Detailed analysis + provenance  │ asynchronous
    │ Auto-fallback on quota          │ polls /jobs/{id}
    └─────────────────────────────────┘
```

**Why 2 tiers?** Because latency matters. Judges (and users) want instant feedback. The FastReply gives it. The PolishReply adds depth. Both cite their sources.

## Core: The Vision Pipeline

1. **Webcam → Chunks**: MediaRecorder captures 2s WebM chunks
2. **Growing File**: Chunks append to a growing WebM file (handles fragmented containers)
3. **YOLO Detection**: Singleton `vision_worker.py` runs YOLOv8n on every frame, returns bounding boxes `[x1,y1,x2,y2]` with labels and confidence
4. **Canvas Overlay**: Boxes are drawn on an HTML Canvas layer over the live video, color-coded per label
5. **Metrics Dashboard**: 6 real-time metrics — chunks, avg latency, P90, model FPS, frames, objects

## Technical Decisions

- **Singleton YOLO worker** — avoids reloading the model on every chunk. Huge latency win.
- **Thread-safe metrics store** — per-stream metrics tracked with locks, exposed via `/stream_status`.
- **LLM provider abstraction** — Gemini + OpenAI with fail-fast on quota errors (429/403). No more infinite hangs.
- **Async job pattern** — notes generation and LLM polish run in background threads, polled via `/jobs/{id}`.
- **Provenance** — every agent response tells you where it got its info (detection data, transcript, notes).

## Performance

| Metric | Value |
|---|---|
| FastReply | <500ms |
| YOLO per-frame | ~50-200ms |
| Stream chunk E2E | ~2-5s |
| PolishReply | ~3-8s |

## What I Learned

- **Latency perception > raw speed.** The 2-tier pattern makes the app *feel* fast even when the LLM takes 5 seconds.
- **Bounding boxes change everything.** A bare label list says "person detected." A canvas overlay makes judges go "wow."
- **Fail-fast on LLM errors.** The early versions hung forever on quota errors. Now they fail in <1 second and fallback gracefully.

## Thanks

Built for the **WeMakeDevs Vision Possible Hackathon** — powered by Vision Agents SDK by Stream, Gemini, and OpenAI.
