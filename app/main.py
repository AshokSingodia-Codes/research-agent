from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from starlette.staticfiles import StaticFiles

from app.api.metrics import router as metrics_router
from app.api.routes import router as research_router
from app.config import settings
from app.persistence.redis_store import redis_store
from app.persistence.worker import worker

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Production security validation
    if settings.env.lower() == "production":
        if not settings.api_key or settings.api_key.startswith("your_") or settings.api_key == "test-key-12345":
            raise RuntimeError("CRITICAL: Production deployment requires a secure, non-default API_KEY.")

    # Startup: resume any incomplete or crashed workflows
    await worker.resume_interrupted_runs()
    yield
    # Shutdown
    redis_store.flush()


app = FastAPI(
    title="Multi-Agent Research Assistant API",
    description="Production-grade multi-agent research platform with LangGraph, Redis durable checkpoints, hard budget enforcement, and step-level tracing.",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS Middleware with environment-driven allowlist
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount Routers
app.include_router(research_router)
app.include_router(metrics_router)

# Mount Static UI assets
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/", response_class=FileResponse, tags=["Dashboard"])
async def serve_dashboard():
    """Serves the minimal single-page HTML viewer dashboard."""
    index_file = STATIC_DIR / "index.html"
    if index_file.exists():
        return FileResponse(str(index_file))
    return HTMLResponse("<h1>Multi-Agent Research Assistant API is running</h1>")
