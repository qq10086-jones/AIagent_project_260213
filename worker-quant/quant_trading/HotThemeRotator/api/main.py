"""FastAPI entry — dashboard JSON + manual-record APIs + static frontend."""
from __future__ import annotations

import os
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
            "JSON over decision_log / calibration / scanner (ADR-0004) plus "
            "manual portfolio record endpoints (P10-23). Rule 3: no "
            "broker/order execution endpoints."
        ),
        version="0.1.0",
    )

    # P22-02 / Rule 15.9 — Remote Personal Access token gate. Active ONLY when
    # HTR_ACCESS_TOKEN is set. Without a token the app itself enforces
    # loopback-only serving (LoopbackOnlyGuard — security review 2026-07-06 C2:
    # the fail-closed rule must hold even when the operator launches raw
    # uvicorn with a non-loopback --host, bypassing tools/serve_remote.py).
    access_token = os.environ.get("HTR_ACCESS_TOKEN")
    if access_token:
        from api.auth import AccessTokenMiddleware

        app.add_middleware(AccessTokenMiddleware, token=access_token)
    else:
        from api.auth import LoopbackOnlyGuard

        app.add_middleware(LoopbackOnlyGuard)

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

    from api.calibration import router as calibration_router  # noqa: E402
    from api.candidate_history import router as candidate_history_router  # noqa: E402
    from api.event_desk import router as event_desk_router  # noqa: E402
    from api.notifier import router as notifier_router  # noqa: E402
    from api.portfolio_fill import router as portfolio_router  # noqa: E402
    from api.proposals import router as proposals_router  # noqa: E402
    from api.reflection import router as reflection_router  # noqa: E402
    from api.watchlist import router as watchlist_router  # noqa: E402

    app.include_router(dashboard_router, prefix="/api")
    app.include_router(symbol_router, prefix="/api")
    app.include_router(portfolio_router, prefix="/api")
    app.include_router(proposals_router, prefix="/api")
    app.include_router(reflection_router, prefix="/api")
    app.include_router(watchlist_router, prefix="/api")
    app.include_router(calibration_router, prefix="/api")
    app.include_router(notifier_router, prefix="/api")
    app.include_router(candidate_history_router, prefix="/api")
    app.include_router(event_desk_router, prefix="/api")

    @app.get("/api/health")
    def health() -> dict:
        return {"status": "ok"}

    # Serve the frontend at root. ADR-0004 Phase 1 = zero-build, so the
    # `frontend/` directory is the deployable artifact (no `dist/`).
    if FRONTEND_ROOT.exists():
        app.mount("/", StaticFiles(directory=str(FRONTEND_ROOT), html=True), name="frontend")

    return app


app = create_app()
