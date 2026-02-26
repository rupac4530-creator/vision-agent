# Judge Checklist — Vision Agent v2.0

> Quick reference for hackathon judges evaluating Vision Agent.

---

## 🔗 Links

| Resource | URL |
|----------|-----|
| **GitHub Repo** | [github.com/rupac4530-creator/vision-agent](https://github.com/rupac4530-creator/vision-agent) |
| **Release v2.0.0** | [v2.0.0 Release](https://github.com/rupac4530-creator/vision-agent/releases/tag/v2.0.0) |
| **README** | [Full Documentation](https://github.com/rupac4530-creator/vision-agent#readme) |

---

## ⚡ Quick Run (30 seconds)

```bash
git clone https://github.com/rupac4530-creator/vision-agent.git
cd vision-agent && cp .env.example backend/.env
# Add your GEMINI_API_KEY to backend/.env
pip install -r backend/requirements.txt
cd backend && uvicorn main:app --host 0.0.0.0 --port 8001
# Open http://localhost:8001
```

---

## 📊 Performance Numbers

| Metric | Value |
|--------|-------|
| FastReply latency | < 500ms (YOLO labels + transcript) |
| LLM polished reply | 1–3s (Gemini 2.0 Flash) |
| Frame processing | ~100ms per frame (YOLOv8n) |
| API cold-start | < 2s |
| Server uptime (verified) | Stable 27+ min session |
| API latency (measured) | 28ms average |

---

## 🤖 AI Features (17 Tabs)

| # | Tab | What It Does |
|---|-----|-------------|
| 1 | Upload & Analyze | Video upload → frame extraction → YOLO detection + audio transcription |
| 2 | AI Study Notes | LLM-generated structured notes, formulas, viva questions from video |
| 3 | Live Stream | Real-time webcam → YOLO every 2s + agent reasoning overlay |
| 4 | Ingest URL | Paste YouTube/Vimeo/Twitter URL → auto-download, analyze, generate notes |
| 5 | Agent Chat | 2-tier reply: instant FastReply + polished LLM response with provenance |
| 6 | Interactive Quiz | Auto-generated MCQ + short-answer from video analysis |
| 7 | Pose Coach | Exercise tracking: joint angles, rep counting, voice coaching (squat/pushup) |
| 8 | Security Camera | Threat detection, wanted poster generation, alert logging |
| 9 | AI Characters | 6 unique personas (Aldrich, Coach, Professor, Dr. Nova, Guardian, Alex) |
| 10 | Crowd Monitor | Density estimation, distress detection, safety scoring, heatmap |
| 11 | Gaming Companion | Upload game screenshots → AI strategy advice + voice commentary |
| 12 | Dashboard | LLM cascade status, API latency charts, frame sampler, session stats |
| 13 | EcoWatch | Environmental monitoring with AI analysis |
| 14 | Blindspot | Driving blindspot detection and alerts |
| 15 | Meeting AI | Real-time meeting transcription and summarization |
| 16 | Accessibility | Voice-guided interface for visually impaired users |
| 17 | Demo Mode | Guided walkthrough of all platform features |

---

## 🔄 LLM Cascade (7 Tiers)

```
Tier 1: Gemini 2.0 Flash      ← preferred (free, fast)
Tier 2: GPT-4o-mini            ← commercial fallback
Tier 3: DeepSeek-R1            ← open-source reasoning
Tier 4: Groq (Llama 3.3 70B)  ← ultra-fast inference
Tier 5: Cloudflare Workers AI  ← edge computing
Tier 6: Ollama (local)         ← offline fallback
Tier 7: Extractive Summary     ← zero-API fallback
```

Each tier has health tracking — if a provider fails 3x in a row, it's deprioritized for 5 minutes.

---

## 📦 SDK Integration

- **22 modules** extracted from [GetStream/Vision-Agents](https://github.com/GetStream/Vision-Agents)
- **50/50 verification tests** passing
- **License**: Apache-2.0 (upstream), MIT (this project)
- Attribution preserved in `THIRD_PARTY_LICENSES.md`

---

## 🏋️ Pose Coach Details (for technical judges)

| Exercise | Detection Method | Threshold |
|----------|-----------------|-----------|
| Squat | Knee angle (hip-knee-ankle) | < 100° = down, > 160° = up |
| Pushup | Elbow angle (shoulder-elbow-wrist) | < 90° = down, > 160° = up |
| Lunge | Front knee angle | < 110° = down |

Rep counting uses a state machine: `UP → DOWN → UP` = 1 rep. Voice coach speaks corrections via Web Speech API.

---

## 🔒 Security

- ✅ No API keys in repository (verified scan)
- ✅ `.env` is gitignored (never committed)
- ✅ `.env.example` with placeholders only
- ✅ GitHub secret scanning alert resolved
- ✅ All code uses `os.getenv()` pattern

---

## 👤 Participant

| | |
|---|---|
| **Name** | Bedanta Chatterjee |
| **School** | S.E. Rly Mixed H.S. School (Class 12) |
| **Country** | India 🇮🇳 |
| **First Hackathon** | Yes |
| **GitHub** | [@rupac4530-creator](https://github.com/rupac4530-creator) |
| **LinkedIn** | [Bedanta Chatterjee](https://www.linkedin.com/in/bedanta-chatterjee-6286ba236) |
