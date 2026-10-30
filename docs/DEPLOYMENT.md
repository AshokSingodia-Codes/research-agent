# Deployment Guide for Multi-Agent Research Assistant

This guide covers deployment options for the Multi-Agent Research Assistant across Docker, Container platforms, and Linux/Cloud Virtual Private Servers (VPS).

---

## 1. Environment Configuration

Before deploying, ensure your `.env` file is configured with production-grade settings:

```env
# Runtime Environment
ENV=production
API_KEY=your-secure-production-api-key-here
CORS_ORIGINS=["https://yourdomain.com"]

# LLM Providers
ANTHROPIC_API_KEY=sk-ant-...
GROQ_API_KEY=gsk_...
TAVILY_API_KEY=tvly-...

# Redis & Concurrency
USE_FAKE_REDIS=false
REDIS_URL=redis://redis:6379/0
RUN_WORKER_INPROCESS=false

# Budget Limits
GLOBAL_BUDGET_DOLLARS=20.00
MAX_SUBQUESTIONS=5
MAX_SEARCHES_PER_SUBQUESTION=3
```

---

## 2. Option A: Docker Compose (Single Host / VM)

The project includes an optimized multi-stage [Dockerfile](file:///e:/7thsem/learning/projects/multiagent/Dockerfile) and [docker-compose.yml](file:///e:/7thsem/learning/projects/multiagent/docker-compose.yml).

### Services Architecture:
1. **`redis`**: Redis 7.2 Alpine persistence store with AOF enabled.
2. **`api`**: FastAPI Web & SSE API running on port `8000`.
3. **`worker`**: Python background stream worker (`app.worker`) consuming queue jobs and executing LangGraph research nodes.

### Commands:

```bash
# 1. Build and run all services in background
docker compose up -d --build

# 2. View streaming logs
docker compose logs -f

# 3. Check health status
docker compose ps

# 4. Stop all services cleanly
docker compose down
```

---

## 3. Option B: Cloud Container Deployment (Render / Railway / Fly.io / AWS ECS)

### Architecture on Cloud:
Deploy two separate services and a managed Redis instance:

1. **Managed Redis**:
   - Provision a Redis instance (Upstash, Redis Cloud, or Cloud Provider Managed Redis).
   - Copy the connection URL (`rediss://default:password@host:port`).

2. **Web Service (API)**:
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn app.main:app --host 0.0.0.0 --port $PORT --workers 2`
   - **Env Vars**: Set `REDIS_URL`, `API_KEY`, `USE_FAKE_REDIS=false`, `RUN_WORKER_INPROCESS=false`, LLM keys.

3. **Background Worker Service**:
   - **Start Command**: `python -m app.worker`
   - **Env Vars**: Same `REDIS_URL` and API keys.

---

## 4. Option C: Linux VPS Deployment (Ubuntu / Debian Systemd)

If deploying directly to an Ubuntu/Debian server without Docker:

### 1. Install Redis & Python
```bash
sudo apt update && sudo apt install -y python3-pip python3-venv redis-server curl
sudo systemctl enable --now redis-server
```

### 2. Set Up Application
```bash
cd /opt
sudo git clone <your-repo-url> multiagent
cd multiagent
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
nano .env # Configure secrets
```

### 3. Systemd Service for API (`/etc/systemd/system/research-api.service`)
```ini
[Unit]
Description=Research Agent FastAPI Server
After=network.target redis-server.service

[Service]
Type=simple
User=www-data
WorkingDirectory=/opt/multiagent
EnvironmentFile=/opt/multiagent/.env
ExecStart=/opt/multiagent/.venv/bin/uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4
Restart=always

[Install]
WantedBy=multi-user.target
```

### 4. Systemd Service for Stream Worker (`/etc/systemd/system/research-worker.service`)
```ini
[Unit]
Description=Research Agent Stream Worker
After=network.target redis-server.service

[Service]
Type=simple
User=www-data
WorkingDirectory=/opt/multiagent
EnvironmentFile=/opt/multiagent/.env
ExecStart=/opt/multiagent/.venv/bin/python -m app.worker
Restart=always

[Install]
WantedBy=multi-user.target
```

### 5. Enable and Start Services
```bash
sudo systemctl daemon-reload
sudo systemctl enable --now research-api research-worker
sudo systemctl status research-api research-worker
```

---

## 5. Post-Deployment Verification

Check endpoints:
- **Healthcheck**: `GET /health` -> `{"status":"ok"}`
- **Metrics**: `GET /metrics` -> Prometheus metrics counters for runs, costs, and citations
- **Web UI**: Navigate to `http://<your-host>:8000/`
