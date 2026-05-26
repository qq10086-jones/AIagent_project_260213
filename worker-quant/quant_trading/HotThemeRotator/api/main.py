"""FastAPI entry — read-only dashboard JSON + (in prod) static frontend mount."""
from __future__ import annotations

import sys
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
# ADR-0004 Phase 1 — zero-build: frontend is plain static (index.html + JSX +
# data.js loaded via CDN React + Babel-standalone). No `dist/` subdirectory;
# FastAPI mounts the `frontend/` directory directly.
FRONTEND_ROOT = PROJECT_ROOT / "frontend"

# Ensure `hot_theme_rotator.*` imports work whether uvicorn is launched from
# project root or somewhere else.
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from api.dashboard import router as dashboard_router  # noqa: E402
from api.symbol import router as symbol_router  # noqa: E402


def create_app() -> FastAPI:
    app = FastAPI(
        title="HotThemeRotator Dashboard API",
        description=(
            "Read-only JSON over decision_log / calibration / scanner (ADR-0004). "
            "Rule 3: no POST/PUT/DELETE, no execution endpoints."
        ),
        version="0.1.0",
    )

    # CORS — Vite dev server default is :5173; prod (same-origin) doesn't need this.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost:5173",
            "http://127.0.0.1:5173",
        ],
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    from api.portfolio_fill import router as portfolio_router  # noqa: E402

    app.include_router(dashboard_router, prefix="/api")
    app.include_router(symbol_router, prefix="/api")
    app.include_router(portfolio_router, prefix="/api")

    @app.get("/api/health")
    def health() -> dict:
        return {"status": "ok"}

    # Serve the frontend at root. ADR-0004 Phase 1 = zero-build, so the
    # `frontend/` directory is the deployable artifact (no `dist/`).
    if FRONTEND_ROOT.exists():
        app.mount("/", StaticFiles(directory=str(FRONTEND_ROOT), html=True), name="frontend")

    return app


app = create_app()
