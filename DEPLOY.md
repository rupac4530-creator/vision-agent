# Deployment Guide — Vision Agent

> Deploy Vision Agent to the cloud in under 10 minutes.

---

## Option 1: Railway (Recommended for Backend)

### One-Click Deploy

[![Deploy on Railway](https://railway.app/button.svg)](https://railway.app/template/vision-agent?referralCode=vision-agent)

### Manual Setup

1. **Create account** at [railway.app](https://railway.app)
2. **New Project** → "Deploy from GitHub repo"
3. **Connect** `rupac4530-creator/vision-agent`
4. **Add Variables** (Settings → Variables):
   ```
   GEMINI_API_KEY=your_key_here
   PORT=8000
   ```
5. Railway auto-detects `railway.json` and deploys
6. Your backend URL: `https://vision-agent-production.up.railway.app`

### Via CLI
```bash
npm install -g @railway/cli
railway login
railway init
railway up
```

---

## Option 2: Render (Free Tier Available)

1. **Create account** at [render.com](https://render.com)
2. **New** → **Web Service** → Connect GitHub repo
3. **Settings**:
   - Runtime: Docker
   - Dockerfile Path: `./Dockerfile`
   - Health Check Path: `/health`
4. **Environment** → Add `GEMINI_API_KEY`
5. **Create Web Service** → auto-deploys
6. Your URL: `https://vision-agent.onrender.com`

> **Note**: Render free tier sleeps after 15 min inactivity. First request takes ~30s to wake.

---

## Option 3: Vercel (Frontend Only)

Best for hosting just the static frontend (HTML/CSS/JS):

1. **Create account** at [vercel.com](https://vercel.com)
2. **Import** → Connect GitHub repo
3. **Settings**:
   - Framework: Other
   - Root Directory: `backend/static`
   - Build Command: (leave empty)
   - Output Directory: `.`
4. **Environment Variables**: Add `NEXT_PUBLIC_API_URL` = your Railway/Render backend URL
5. **Deploy**

---

## Option 4: Docker (Self-Hosted / VPS)

```bash
# Clone and build
git clone https://github.com/rupac4530-creator/vision-agent.git
cd vision-agent

# Set up environment
cp .env.example backend/.env
# Edit backend/.env and add your GEMINI_API_KEY

# Run with Docker Compose (includes Ollama)
docker-compose up -d

# Or just the backend
docker build -t vision-agent .
docker run -p 8000:8000 --env-file backend/.env vision-agent
```

---

## Option 5: Local Development

```bash
git clone https://github.com/rupac4530-creator/vision-agent.git
cd vision-agent
cp .env.example backend/.env
# Edit backend/.env — add GEMINI_API_KEY

pip install -r backend/requirements.txt
cd backend
uvicorn main:app --host 0.0.0.0 --port 8001 --reload
# Open http://localhost:8001
```

---

## Environment Variables Reference

| Variable | Required | Description |
|----------|----------|-------------|
| `GEMINI_API_KEY` | ✅ Yes | Google Gemini 2.0 Flash API key |
| `OPENAI_API_KEY` | Optional | OpenAI key (Tier 2 fallback) |
| `GROQ_API_KEY` | Optional | Groq key (Tier 4 ultra-fast) |
| `PORT` | Auto | Set by Railway/Render automatically |
| `OLLAMA_HOST` | Optional | Ollama URL (default: `http://localhost:11434`) |

---

## Verification

After deployment, verify these endpoints:

```bash
# Health check
curl https://YOUR_URL/health
# Expected: {"status": "ok", "llm_display": "GLM-5 (glm-5:cloud)", ...}

# Models
curl https://YOUR_URL/models
# Expected: {"providers": [...], "active": "gemini-2.0-flash"}

# API docs
open https://YOUR_URL/docs
```

---

## CI/CD Pipeline

Pushes to `main` automatically trigger:
1. **CI** (`.github/workflows/ci.yml`): Lint + test across Python 3.10/3.11/3.12
2. **Deploy** (`.github/workflows/deploy.yml`): Build Docker → Push to GHCR → Deploy to Railway

### Setting Up Auto-Deploy

1. Go to **GitHub → Settings → Secrets → Actions**
2. Add these secrets:
   - `RAILWAY_TOKEN` — from Railway dashboard → Account → Tokens
   - `GEMINI_API_KEY` — your Google API key
3. Push to `main` → auto-deploy triggers

---

## Estimated Costs

| Platform | Plan | Cost | Notes |
|----------|------|------|-------|
| Railway | Starter | $5/mo credit free | Enough for demo |
| Render | Free | $0 | Sleeps after 15 min |
| Vercel | Hobby | $0 | Frontend only |
| Docker VPS | Any | $5-20/mo | Full control |
