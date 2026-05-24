"""GET /api/dashboard — V3 dashboard JSON shape (ADR-0004)."""
from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter

from api.serializers import build_dashboard_payload


router = APIRouter()

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@router.get("/dashboard")
def get_dashboard(top_n: int = 10) -> dict:
    """Return the full V3 dashboard payload.

    Read-only — Python data layer is the single source of truth (ADR-0003 +
    ADR-0004). Sources:
    - `gates` from `_GATE_DEFINITIONS` (P9-01 / P9-02 / P9-03 truth)
    - `candidates` from `build_sample_panel` (Streamlit sample path; yfinance
      mode is the user's runtime choice and lands here when wired)
    - `decisionLog` from `read_predictions` for the current trade date
    - `meta.calibration` from `build_calibration_badge` — kept at
      `insufficient_calibration` until P9-03 has accumulated 100+ paired samples
    - `markets` / `themes` / `newsTimeline` / `kline` return as empty lists
      (Python layer doesn't yet produce these; frontend renders "数据未就绪")
    """
    return build_dashboard_payload(base_dir=PROJECT_ROOT, top_n=int(top_n))
