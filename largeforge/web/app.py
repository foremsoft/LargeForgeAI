"""Main FastAPI application for the web interface."""

import os
from pathlib import Path
from typing import Optional

import uvicorn
from fastapi import FastAPI, Request, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from largeforge.config.web import WebConfig
from largeforge.utils import get_logger
from largeforge.version import __version__
from largeforge.web import api
from largeforge.web.state import JobStateManager
from largeforge.web.websocket import (
    connection_manager,
    global_websocket_endpoint,
    websocket_endpoint,
)

logger = get_logger(__name__)


def create_app(config: Optional[WebConfig] = None) -> FastAPI:
    """
    Create and configure the FastAPI application.

    Args:
        config: Web configuration

    Returns:
        Configured FastAPI app
    """
    if config is None:
        config = WebConfig()

    # Create FastAPI app
    app = FastAPI(
        title="LargeForgeAI",
        description="Web interface for LLM training and deployment",
        version=__version__,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # Store config in app state
    app.state.config = config
    app.state.state_manager = None

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.cors_origins,
        allow_credentials=config.cors_allow_credentials,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include API router
    app.include_router(api.router)

    # WebSocket routes
    @app.websocket("/ws/jobs/{job_id}")
    async def job_websocket(websocket: WebSocket, job_id: str):
        """WebSocket endpoint for job-specific updates."""
        state_manager = app.state.state_manager
        if state_manager:
            await websocket_endpoint(websocket, job_id, state_manager)

    @app.websocket("/ws")
    async def global_ws(websocket: WebSocket):
        """Global WebSocket endpoint for all updates."""
        await global_websocket_endpoint(websocket)

    # Startup event
    @app.on_event("startup")
    async def startup():
        """Initialize resources on startup."""
        logger.info(f"Starting LargeForgeAI Web Server v{__version__}")

        # Initialize state manager
        state_manager = JobStateManager(config.job_storage_path)
        app.state.state_manager = state_manager
        api.set_state_manager(state_manager)

        logger.info(f"Job storage: {config.job_storage_path}")
        logger.info(f"Loaded {state_manager.count_jobs()} existing jobs")

    # Shutdown event
    @app.on_event("shutdown")
    async def shutdown():
        """Cleanup resources on shutdown."""
        logger.info("Shutting down LargeForgeAI Web Server")

    # Exception handlers
    @app.exception_handler(ValueError)
    async def value_error_handler(request: Request, exc: ValueError):
        return JSONResponse(
            status_code=400,
            content={"error": "Validation error", "detail": str(exc)},
        )

    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception):
        logger.error(f"Unhandled exception: {exc}")
        return JSONResponse(
            status_code=500,
            content={"error": "Internal server error", "detail": str(exc)},
        )

    # Serve static files (React frontend)
    if config.serve_frontend and config.static_files_path:
        static_path = Path(config.static_files_path)
        if static_path.exists():
            app.mount("/static", StaticFiles(directory=str(static_path)), name="static")

            # Catch-all route for SPA routing
            @app.get("/{full_path:path}")
            async def serve_spa(full_path: str):
                """Serve the React SPA for all unmatched routes."""
                # Skip API and WebSocket routes
                if full_path.startswith(("api/", "ws/", "docs", "redoc", "openapi.json")):
                    return JSONResponse(
                        status_code=404,
                        content={"error": "Not found"},
                    )

                index_path = static_path / "index.html"
                if index_path.exists():
                    return FileResponse(str(index_path))

                return JSONResponse(
                    status_code=404,
                    content={"error": "Frontend not found"},
                )

    # Root endpoint
    @app.get("/")
    async def root():
        """Root endpoint with API info."""
        return {
            "name": "LargeForgeAI",
            "version": __version__,
            "docs": "/docs",
            "api": "/api/v1",
        }

    return app


def run_server(
    host: str = "0.0.0.0",
    port: int = 7860,
    config: Optional[WebConfig] = None,
    reload: bool = False,
    workers: int = 1,
) -> None:
    """
    Run the web server.

    Args:
        host: Host to bind to
        port: Port to bind to
        config: Web configuration
        reload: Enable auto-reload for development
        workers: Number of worker processes
    """
    if config is None:
        config = WebConfig(host=host, port=port)
    else:
        config.host = host
        config.port = port

    app = create_app(config)

    logger.info(f"Starting server at http://{host}:{port}")
    logger.info(f"API docs at http://{host}:{port}/docs")

    uvicorn.run(
        app,
        host=host,
        port=port,
        reload=reload,
        workers=workers if not reload else 1,
        log_level="info",
    )


# For running with uvicorn directly: uvicorn largeforge.web.app:app
app = create_app()
