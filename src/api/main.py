import os
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from starlette.staticfiles import StaticFiles

from src.api.routes import router as research_router
from src.config import settings

STATIC_DIR = Path(__file__).resolve().parent.parent.parent / "static"

app = FastAPI(
    title="Multi-Agent Research Assistant API",
    description="Production-grade multi-agent research system with LangGraph, Redis durable checkpoints, hard budget enforcement, and step-level tracing.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static assets
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# Mount API routes
app.include_router(research_router)


@app.get("/health", tags=["Health"])
async def health_check():
    """Health status and configuration inspection."""
    return {
        "status": "healthy",
        "active_llm_provider": settings.active_llm_provider,
        "mock_llm": settings.is_mock_llm,
        "mock_search": settings.is_mock_search,
        "groq_model": settings.groq_model if settings.has_groq else None,
        "use_fake_redis": settings.use_fake_redis,
        "max_subquestions": settings.max_subquestions,
        "max_revision_loops": settings.max_revision_loops,
    }


@app.get("/", response_class=FileResponse, tags=["Dashboard"])
async def serve_dashboard():
    """Serves the rich Tailwind CSS + JavaScript interactive dashboard."""
    index_file = STATIC_DIR / "index.html"
    if index_file.exists():
        return FileResponse(str(index_file))
    return HTMLResponse("<h1>Multi-Agent Research Assistant API is running</h1>")
