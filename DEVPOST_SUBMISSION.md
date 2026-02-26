# DEVPOST SUBMISSION — Copy Each Section Into The Form

---

## Project Name

```
Vision Agent — Real-Time Multimodal AI Platform
```

---

## Tagline (one-liner)

```
A camera with a brain: real-time YOLO detection, fitness coaching with 7 exercises, 2-tier agent intelligence, and voice feedback — all under 500ms.
```

---

## What It Does

```
Vision Agent is an all-in-one real-time multimodal AI platform that transforms any camera into an intelligent observer, coach, and advisor.

Core Capabilities:
- Live webcam/screen streaming with instant YOLOv8 object detection and bounding box overlays
- Fitness coaching across 7 exercises (squat, push-up, lunge, plank, jumping jack, shoulder press, auto-detect) with rep counting, streak tracking, form scoring, and browser voice feedback
- 2-tier agent intelligence: instant FastReply (<500ms) from YOLO labels + transcript, plus deep LLM PolishReply with provenance links
- Video upload with full pipeline: frame extraction → audio transcription (Whisper) → object detection → AI notes → interactive quiz
- URL ingestion from YouTube/Vimeo/Twitter with auto-download and analysis
- Real-time metrics dashboard: chunks processed, avg/P90 latency, model FPS, total detections
- Screen share support for analyzing any on-screen content
- Audio visualization, session timing, and confetti celebrations on milestones
- Export reports as JSON or Markdown
- Keyboard shortcuts for power users

The platform uses a cascade LLM architecture (Gemini → OpenAI → Cloudflare Workers AI → local fallback) ensuring it works even without API keys.
```

---

## How We Built It

```
Backend:
- Python 3.12 + FastAPI for the REST API
- YOLOv8 (Ultralytics) for real-time object detection with bounding box coordinates
- MediaPipe Pose for skeletal pose estimation and joint angle computation
- Custom FitnessCoach engine with state machines for 7 exercises
- OpenAI Whisper for audio transcription
- Cascade LLM provider: Gemini 2.0 Flash → OpenAI GPT-4o-mini → Cloudflare Workers AI → local extractive summary
- Thread-safe async job system for background LLM tasks
- Growing-file WebM strategy for robust browser chunk streaming
- ffmpeg for video transcoding and audio extraction
- yt-dlp for URL video ingestion

Frontend:
- Single-page vanilla HTML/CSS/JS (zero framework dependencies)
- Canvas API for real-time bounding box overlays on video
- Web Audio API for live audio visualization
- MediaRecorder API for 2-second chunk streaming
- SpeechSynthesis API for voice coaching feedback
- 3D card tilt effects, animated gradient text, confetti particle engine
- Cosmic starfield background with breathing nebulae and shooting stars
- Glassmorphism dark theme with smooth micro-interactions

Architecture:
- Vision Agents SDK patterns for real-time video pipeline
- 2-tier agent reasoning (FastReply + PolishReply) with provenance tracking
- Singleton YOLO worker with latency tracking (avg/P90/FPS)
- WebRTC-inspired chunk streaming with growing-file WebM decoding
```

---

## Challenges We Ran Into

```
1. WebM Chunk Decoding: Browser MediaRecorder produces fragmented WebM where only chunk 1 has the EBML header. Subsequent chunks are bare clusters that ffmpeg can't decode alone. Solution: a growing-file strategy that maintains a complete WebM by appending each chunk to the init segment.

2. Sub-500ms Agent Response: Getting meaningful AI responses in under 500ms required a 2-tier architecture — deterministic FastReply from cached YOLO labels + template matching, while the full LLM runs in the background.

3. Pose Estimation Accuracy: MediaPipe joint angles can be noisy. We implemented best-side selection (comparing left/right) and threshold-based state machines for reliable rep counting across 7 different exercises.

4. LLM Quota Management: Free-tier APIs hit rate limits. Built a cascade provider that gracefully falls through Gemini → OpenAI → Cloudflare → local summary without any user-visible error.

5. Cross-browser Video: Different browsers support different MediaRecorder codecs. Implemented format detection with VP8/VP9 fallbacks and multi-codec transcoding.
```

---

## Accomplishments We're Proud Of

```
- 7 exercise types with real-time rep counting, streak tracking, and motivational voice feedback
- True real-time object detection with bounding box overlays rendered on HTML5 Canvas
- 2-tier agent that responds in <500ms deterministically, then enhances with full LLM analysis
- Zero-dependency frontend (no React/Vue/Angular) — pure HTML/CSS/JS with premium animations
- Works offline: graceful fallback to local models when cloud APIs are unavailable
- Confetti celebrations when you hit rep milestones
- Screen share support for analyzing any on-screen content
- Premium UI with 3D card tilt, animated gradients, shooting stars, and breathing nebulae
```

---

## What We Learned

```
- Real-time AI requires thinking in pipelines, not request-response
- The transport layer (WebRTC/chunking) matters as much as the AI models
- Free-tier APIs can be powerful when strategically cascaded
- Browser APIs (MediaRecorder, SpeechSynthesis, Web Audio, Canvas) are incredibly capable
- A polished UI dramatically improves perception of AI capability
- Pose estimation + heuristic coaching can be surprisingly effective without heavy ML
```

---

## What's Next

```
- WebRTC native integration with Vision Agents SDK Edge transport
- Multi-person tracking and coaching
- Custom YOLO model training for sport-specific equipment detection
- Mobile app using React Native SDK
- Real-time avatar generation from pose data
- Persistent session history with trend analysis
- Deployment to Railway (backend) + Vercel (frontend)
```

---

## Built With (Tags)

```
python, fastapi, yolo, ultralytics, mediapipe, opencv, whisper, gemini, openai, cloudflare-workers-ai, webrtc, canvas-api, speech-synthesis, javascript, html5, css3, ffmpeg, docker, vision-agents-sdk
```

---

## Links

```
GitHub: https://github.com/rupac4530-creator/vision-agent
Demo: http://localhost:8000 (run locally)
```

---

## Video Demo URL

```
[Upload your 2-minute demo video to YouTube/Loom and paste URL here]
```

---

## Category

```
Vision Possible: Agent Protocol Hackathon — by WeMakeDevs + Stream
```
