# Building a Real-Time Lecture Assistant with Vision Agents and LLMs

Lecture recordings are everywhere, but turning them into useful study material is still time-consuming. For the **WeMakeDevs Vision Possible hackathon**, I built **Vision Agent** — a real-time multimodal system that ingests video, extracts audio and frames, and produces concise notes, LaTeX-rendered formulas, and viva-style questions to help students revise faster.

## Why this matters

Many students struggle to convert long lectures into exam-ready notes. Vision Agent automates the heavy lifting while preserving **provenance**: every key fact is linked to a transcript excerpt and a video frame, so teachers and students can verify and trust the outputs.

## How it works

```
Video Upload → Frame Extraction → Audio Extraction
    ↓                ↓                  ↓
 Thumbnails     YOLOv8 Labels    Whisper Transcript
    ↓                ↓                  ↓
              Multimodal Context
                     ↓
            LLM (GPT-4o-mini)
                     ↓
       Notes + Formulas + Viva Qs + Quiz
```

### 1. Chunked Streaming

The demo streams 2–5s video chunks for low-latency feedback, using a lightweight local pipeline for fast responses. Each chunk returns a transcript snippet and top visual labels immediately.

### 2. Multimodal Analysis

For each chunk, audio is transcribed (Whisper for latency) and frames are sampled for vision labels (YOLOv8n). This creates synchronized multimodal context.

### 3. LLM Synthesis

A deterministic LLM prompt (temperature 0) turns transcript + frame labels into a final study package: summary, concepts, formulas (LaTeX), and graded viva questions (easy/medium/hard).

### 4. Interactive Features (Day 5)

- **Ask a Question**: Contextual QA powered by the same LLM, answering from the transcript and notes
- **Quiz Generator**: Creates MCQs with plausible distractors and short-answer questions
- **Timeline Navigation**: Clickable thumbnails jump to specific points in the lecture
- **Formula Rendering**: MathJax renders LaTeX formulas inline

## What I learned

- **Latency vs. quality trade-offs matter.** Using tiny/smaller models gives great interactivity for streaming chunks, but a final pass with stronger models improves quality for the polished notes.
- **Provenance is essential.** Judges and users trust outputs more when claims are backed by transcript snippets and frame images.
- **A clean demo and clear metrics** (per-chunk latency, model names, and sample outputs) make the difference in hackathon submissions.

## Demo highlights

- Per-chunk latency: **~2–5s** (whisper + yolov8n on laptop)
- Outputs: summary, formulas (rendered), 10 viva questions (easy→hard), and an interactive quiz generator
- Interactive QA chat: ask anything about the lecture content

## Next steps

I'd like to extend Vision Agent to run on the edge (Stream Vision Agents SDK), add speaker diarization for multi-speaker lectures, and integrate better formula extraction (LaTeX OCR for whiteboard content).

## Thanks

Huge thanks to the **WeMakeDevs** community and the **Vision Agents SDK by Stream** for the tools and inspiration. And to **OpenAI** for the powerful models that make multimodal understanding practical. The code and demo are open on GitHub — link in the submission.

---

*Built for the WeMakeDevs Vision Possible: Agent Protocol Hackathon*
