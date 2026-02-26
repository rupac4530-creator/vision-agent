# DEMO VIDEO SCRIPT (2 Minutes)

Record with OBS Studio or any screen recorder at 1080p.

---

## BEFORE RECORDING

1. Open terminal, navigate to `vision-agent/backend`
2. Run: `python main.py`
3. Open browser at `http://localhost:8000`
4. Make sure webcam is connected
5. Have the Live Stream tab ready
6. Test one quick stream chunk to warm up the YOLO model

---

## 0:00 — 0:10 | HOOK (Show the Problem)

**[Show the platform homepage]**

> "What if your camera could think? Vision Agent is a real-time multimodal AI platform that watches, listens, detects, and reasons — all in under 500 milliseconds."

**[Click through tabs quickly to show all 6 tabs]**

---

## 0:10 — 0:40 | LIVE STREAM DEMO (The Wow Factor)

**[Switch to Live Stream tab]**

1. Select "Squat" from exercise dropdown
2. Click "Start Stream"
3. **Stand up and do 3 squats in front of the camera**

> "Watch the real-time detections — YOLO identifies me instantly with bounding boxes. The fitness coach is counting my reps, scoring my form, and giving voice feedback."

**[Point to the metrics: chunks, latency, objects, frames]**

> "Every 2-second chunk is processed in real-time. You can see avg latency, P90, and model FPS right here."

**[Point to the coach panel showing reps and cue]**

---

## 0:40 — 1:00 | SCREEN SHARE + DETECTION

**[Click "Screen" button to switch to screen share]**

> "It's not just cameras — I can share my screen too. The AI detects objects in any video content on screen."

**[Show a YouTube video or image on screen, let detection run for a few seconds]**

**[Switch back to camera with "Camera" button]**

---

## 1:00 — 1:20 | AGENT INTELLIGENCE

**[Click on a detected label in the live labels area]**

> "Click any detected object to ask the AI. It responds in two tiers — an instant FastReply under 500 milliseconds, then a deep LLM analysis arrives seconds later with full provenance."

**[Show the green FastReply badge, then the purple PolishReply appearing]**

**[Switch to Agent Chat tab, type a question]**

> "The agent chat works the same way — instant detection-based answer, then full AI reasoning."

---

## 1:20 — 1:40 | FULL PIPELINE

**[Switch to Upload & Analyze tab]**

> "For recorded videos, the full pipeline runs: frame extraction, Whisper transcription, YOLO detection, and AI-generated study notes with formulas and quiz questions."

**[If you have a pre-analyzed video, show the results: metrics, transcript, detections]**

**[Switch to AI Notes tab, show notes if available]**

---

## 1:40 — 2:00 | ARCHITECTURE & CLOSING

> "Under the hood: FastAPI backend with YOLOv8, MediaPipe Pose for 7 exercises, a cascade LLM system that gracefully falls through Gemini, OpenAI, Cloudflare, and local summary. The frontend is zero-dependency — pure HTML, CSS, and JavaScript with WebRTC-inspired streaming."

> "Vision Agent: a camera with a brain. Built for the Vision Possible hackathon with the Vision Agents SDK."

**[Show the footer with tech stack, end on the animated logo]**

---

## RECORDING TIPS

- Record at 1080p, 30fps minimum
- Keep your face visible during exercise demo (proves it's real-time)
- Make sure the metrics dashboard is visible during streaming
- Speak clearly and at moderate pace
- Upload to YouTube as Unlisted, paste link in Devpost
- Total: exactly 2 minutes or under
