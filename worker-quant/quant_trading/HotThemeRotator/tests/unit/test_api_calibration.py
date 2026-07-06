"""API tests for /api/calibration/reliability (P3.6)."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(monkeypatch):
    base = Path(".runtime") / "calibration_api_tests"
    if base.exists():
        shutil.rmtree(base, ignore_errors=True)
    base.mkdir(parents=True, exist_ok=True)
    import api.calibration as cal_mod
    monkeypatch.setattr(cal_mod, "PROJECT_ROOT", base)
    from api.main import create_app
    yield TestClient(create_app()), base
    shutil.rmtree(base, ignore_errors=True)


def _write_recalibrator(base):
    (base / "reports").mkdir(parents=True, exist_ok=True)
    (base / "reports" / "recalibrator_isotonic_v1.json").write_text(json.dumps({
        "model_version": "isotonic_v1",
        "fitted_at": "2026-05-28T00:00:00+00:00",
        "evidence_origin": "bootstrap",
        "sample_count": 762,
        "horizon_days": 3,
        "trade_date_range": ["2026-03-23", "2026-04-13"],
        "breakpoints": [
            {"x_min": 0.5, "x_max": 0.85, "y_hat": 0.48, "n": 500},
            {"x_min": 0.85, "x_max": 0.95, "y_hat": 0.63, "n": 200},
        ],
    }), encoding="utf-8")


def _write_kfold(base, verdict="downgrade"):
    (base / "reports").mkdir(parents=True, exist_ok=True)
    (base / "reports" / "recalibrator_kfold_v1.json").write_text(json.dumps({
        "verdict": {
            "verdict": verdict,
            "reason": "test reason",
            "oos_brier_mean": 0.28,
            "in_sample_brier": 0.24,
            "random_baseline": 0.25,
        },
        "report": {
            "oos_brier_std": 0.05,
            "oos_brier_min": 0.24,
            "oos_brier_max": 0.41,
            "folds": [
                {"fold_idx": 0, "train_n": 600, "test_n": 150, "raw_brier": 0.34, "calibrated_brier": 0.24, "n_blocks": 5},
                {"fold_idx": 1, "train_n": 600, "test_n": 150, "raw_brier": 0.41, "calibrated_brier": 0.41, "n_blocks": 11},
            ],
        },
        "args": {},
    }), encoding="utf-8")


def test_calibration_empty(client):
    c, _ = client
    payload = c.get("/api/calibration/reliability").json()
    assert payload["recalibrator"]["fitted"] is False
    assert payload["kfold"]["available"] is False


def test_calibration_with_recalibrator_only(client):
    c, base = client
    _write_recalibrator(base)
    payload = c.get("/api/calibration/reliability").json()
    assert payload["recalibrator"]["fitted"] is True
    assert payload["recalibrator"]["sample_count"] == 762
    assert len(payload["recalibrator"]["blocks"]) == 2
    assert payload["kfold"]["available"] is False


def test_calibration_with_kfold(client):
    c, base = client
    _write_recalibrator(base)
    _write_kfold(base, verdict="downgrade")
    payload = c.get("/api/calibration/reliability").json()
    assert payload["kfold"]["available"] is True
    assert payload["kfold"]["verdict"] == "downgrade"
    assert payload["kfold"]["oos_brier_mean"] == 0.28
    assert len(payload["kfold"]["folds"]) == 2


def test_calibration_endpoint_is_get_only(client):
    c, _ = client
    assert c.post("/api/calibration/reliability").status_code == 405
