## Vision Agent – Verification Checklist (run 2–3 times)

This is the “nothing left out” checklist to validate **all features** before demo/submission.

### 0) Safety

- Do **NOT** paste API keys into chat/screenshots/logs.
- Put secrets only in `vision-agent/backend/.env` (see `.env.example`).

---

### 1) Install + run backend

From PowerShell:

```powershell
cd "D:\VisionAgent_COMPLETE_SAVE_2026-02-24\vision-agent\backend"
python -m venv venv
.\venv\Scripts\pip install -r requirements.txt
.\venv\Scripts\python.exe -m uvicorn main:app --reload --port 8000
```

Open:
- `http://localhost:8000/health` (must return `"status":"ok"`)
- `http://localhost:8000` (UI loads)

---

### 2) Tab-by-tab feature checks

Run this sequence **2–3 times**.

#### Tab 1: Upload & Analyze
- Upload a short mp4 (10–30s)
- Run **Full Analyze**
- Confirm:
  - transcript shown
  - detections summary shown
  - downloads work (analysis/transcript/detections)

#### Tab 2: AI Notes
- Click **Generate Notes**
- Confirm:
  - job completes
  - notes render (summary, concepts, viva, raw JSON)
  - provider badge shows (Gemini/OpenAI/Cloudflare/local-summary)

#### Tab 3: Live Stream
- Click **Start Stream**
- Confirm:
  - bbox overlay draws
  - live labels appear
  - metrics update
  - **Live Coach** panel updates:
    - rep counter moves when doing squats/push-ups
    - cue text updates
    - voice toggle speaks cues (if enabled)
  - Stop stream → `stream_status` log prints final summary including coach reps

> If rep counting doesn’t work, ensure `mediapipe` installed. The rest of the app should still work.

#### Tab 4: Ingest URL
- Paste a YouTube/Vimeo URL and consent checkbox
- Confirm:
  - job progresses and completes
  - results show

#### Tab 5: Agent Chat
- Ask a question
- Confirm:
  - FastReply appears immediately
  - PolishReply appears later (job polling)

#### Tab 6: Quiz
- Generate quiz
- Confirm:
  - MCQ + short answer render
  - scoring works

---

### 3) Minimal “demo readiness” record

Capture (copy/paste into your submission notes):
- `/health` JSON
- A completed `analysis.json`
- A completed `notes.json`
- A completed `quiz.json`
- A live stream log section showing:
  - chunk processing
  - bbox + coach reps

