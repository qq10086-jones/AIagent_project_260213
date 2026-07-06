"""Calibration observability API (read-only).

GET /api/calibration/reliability — isotonic recalibrator + K-fold verdict
                                    + per-fold Brier breakdown for UI viz.

Rule 9.4 — surfacing this data is transparency, not action. No write paths.
"""
from __future__ import annotations

import json
from pathlib import Path

from fastapi import APIRouter


router = APIRouter()
PROJECT_ROOT = Path(__file__).resolve().parents[1]


@router.get("/calibration/reliability")
def get_calibration_reliability() -> dict:
    """Return isotonic blocks + K-fold report for the calibration viz.

    Shape:
    {
      "recalibrator": {
        "fitted": bool,
        "model_version": str | null,
        "evidence_origin": str | null,
        "sample_count": int | null,
        "horizon_days": int | null,
        "trade_date_range": [str, str] | null,
        "blocks": [{x_min, x_max, y_hat, n}, ...]
      },
      "kfold": {
        "available": bool,
        "verdict": "ship" | "downgrade" | "caution_overfit" | null,
        "reason": str | null,
        "oos_brier_mean": float | null,
        "oos_brier_std": float | null,
        "in_sample_brier": float | null,
        "folds": [{fold_idx, train_n, test_n, raw_brier, calibrated_brier}, ...]
      }
    }
    """
    recal_path = PROJECT_ROOT / "reports" / "recalibrator_isotonic_v1.json"
    recal_payload = None
    if recal_path.exists():
        try:
            recal_payload = json.loads(recal_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            recal_payload = None

    kfold_path = PROJECT_ROOT / "reports" / "recalibrator_kfold_v1.json"
    kfold_payload = None
    if kfold_path.exists():
        try:
            kfold_payload = json.loads(kfold_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            kfold_payload = None

    recal_out = {
        "fitted": recal_payload is not None,
        "model_version": (recal_payload or {}).get("model_version"),
        "evidence_origin": (recal_payload or {}).get("evidence_origin"),
        "sample_count": (recal_payload or {}).get("sample_count"),
        "horizon_days": (recal_payload or {}).get("horizon_days"),
        "trade_date_range": (recal_payload or {}).get("trade_date_range"),
        "blocks": (recal_payload or {}).get("breakpoints", []),
    }

    if kfold_payload is None:
        kfold_out = {"available": False}
    else:
        verdict = kfold_payload.get("verdict", {})
        report = kfold_payload.get("report", {})
        kfold_out = {
            "available": True,
            "verdict": verdict.get("verdict"),
            "reason": verdict.get("reason"),
            "oos_brier_mean": verdict.get("oos_brier_mean"),
            "oos_brier_std": report.get("oos_brier_std"),
            "oos_brier_min": report.get("oos_brier_min"),
            "oos_brier_max": report.get("oos_brier_max"),
            "in_sample_brier": verdict.get("in_sample_brier"),
            "random_baseline": verdict.get("random_baseline", 0.25),
            "folds": [
                {
                    "fold_idx": f["fold_idx"],
                    "train_n": f["train_n"],
                    "test_n": f["test_n"],
                    "raw_brier": f["raw_brier"],
                    "calibrated_brier": f["calibrated_brier"],
                    "n_blocks": f.get("n_blocks"),
                }
                for f in report.get("folds", [])
            ],
        }

    return {"recalibrator": recal_out, "kfold": kfold_out}
