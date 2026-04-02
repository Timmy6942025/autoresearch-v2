"""FastAPI server for AutoResearch v2 — exposes /research, /status, /health endpoints."""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from autoresearch.core.config import ResearchConfig
from autoresearch.core.engine import ResearchEngine
from autoresearch.core.state import StateManager

logger = logging.getLogger("autoresearch.server")

# ─── Request / Response Models ───────────────────────────────────────────────


class ResearchRequest(BaseModel):
    query: str = Field(..., description="Research query")
    depth: str = Field("medium", description="Research depth: brief, medium, deep")
    max_sources: int = Field(20, description="Max sources to analyze")
    output_format: str = Field(
        "markdown", description="Output format: markdown, json, html"
    )
    files: Optional[List[str]] = Field(
        None, description="Local document paths to include"
    )


class ResearchResponse(BaseModel):
    id: str
    status: str
    query: str
    report: Optional[str] = None
    error: Optional[str] = None
    findings_count: int = 0
    elapsed_seconds: float = 0.0


class StatusResponse(BaseModel):
    status: str
    version: str
    active_research: Optional[str] = None
    completed_research: int = 0
    uptime_seconds: float = 0.0


class HealthResponse(BaseModel):
    status: str
    timestamp: float


# ─── Server State ────────────────────────────────────────────────────────────


class ServerState:
    def __init__(self) -> None:
        self.start_time = time.time()
        self.engine: Optional[ResearchEngine] = None
        self.active_task: Optional[str] = None
        self.completed_count = 0
        self.state_manager = StateManager()
        self._lock = asyncio.Lock()


state = ServerState()


# ─── Lifespan ────────────────────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI):
    config = ResearchConfig()
    state.engine = ResearchEngine(config)
    logger.info("Research engine initialized")
    yield
    logger.info("Server shutting down")


# ─── App ─────────────────────────────────────────────────────────────────────


def create_app(enable_cors: bool = True) -> FastAPI:
    app = FastAPI(
        title="AutoResearch v2",
        description="MLX-Powered Autonomous Research Engine",
        version="0.2.0",
        lifespan=lifespan,
    )

    if enable_cors:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=False,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    return app


app = create_app()


# ─── Endpoints ───────────────────────────────────────────────────────────────


@app.post("/research", response_model=ResearchResponse)
async def post_research(req: ResearchRequest) -> ResearchResponse:
    """Submit a research query. Runs synchronously and returns the full report."""
    if not state.engine:
        raise HTTPException(status_code=503, detail="Engine not initialized")

    async with state._lock:
        if state.active_task:
            raise HTTPException(
                status_code=429,
                detail=f"Research already in progress: {state.active_task}",
            )
        research_id = str(uuid.uuid4())[:8]
        state.active_task = research_id

    try:
        config = ResearchConfig()
        config.search.max_results = req.max_sources
        engine = ResearchEngine(config)

        start = time.time()
        report = await engine.research(
            query=req.query,
            depth=req.depth,
            output_format=req.output_format,
            files=req.files,
        )
        elapsed = time.time() - start

        findings = engine.get_state().findings
        state.completed_count += 1

        return ResearchResponse(
            id=research_id,
            status="completed",
            query=req.query,
            report=report,
            findings_count=len(findings),
            elapsed_seconds=round(elapsed, 2),
        )
    except Exception as e:
        logger.error("Research failed: %s", e)
        return ResearchResponse(
            id=research_id,
            status="failed",
            query=req.query,
            error="Research failed. Check server logs for details.",
        )
    finally:
        state.active_task = None


@app.get("/research/{research_id}", response_model=ResearchResponse)
async def get_research(research_id: str) -> ResearchResponse:
    """Get a past research result by ID."""
    result = state.state_manager.load(research_id)
    if not result:
        raise HTTPException(status_code=404, detail=f"Research {research_id} not found")

    return ResearchResponse(
        id=research_id,
        status=result.status,
        query=result.query,
        report=result.report,
        error=result.error,
        findings_count=len(result.findings),
        elapsed_seconds=round(result.completed_at - result.started_at, 2)
        if result.completed_at
        else 0.0,
    )


@app.get("/status", response_model=StatusResponse)
async def get_status() -> StatusResponse:
    """Get server status and active research info."""
    return StatusResponse(
        status="running",
        version="0.2.0",
        active_research=state.active_task,
        completed_research=state.completed_count,
        uptime_seconds=round(time.time() - state.start_time, 1),
    )


@app.get("/health", response_model=HealthResponse)
async def get_health() -> HealthResponse:
    """Health check endpoint."""
    return HealthResponse(status="healthy", timestamp=time.time())


@app.get("/models")
async def list_models() -> Dict[str, Any]:
    """List available models."""
    from autoresearch.core.models import registry

    return {
        "models": [
            {
                "name": m.name,
                "path": m.path,
                "params_m": m.params_m,
                "max_context": m.max_context,
                "supports_turboquant": m.supports_turboquant,
            }
            for m in registry.list_all()
        ]
    }


# ─── CLI Integration ─────────────────────────────────────────────────────────


def run_server(
    host: str = "127.0.0.1",
    port: int = 8080,
    enable_cors: bool = True,
) -> None:
    """Run the FastAPI server via uvicorn."""
    import uvicorn

    uvicorn.run(
        "autoresearch.server:app",
        host=host,
        port=port,
        reload=False,
        log_level="info",
    )
