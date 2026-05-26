"""P10-13 Backdated Calibration Bootstrap tests (ADR-0006)."""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from hot_theme_rotator.calibration.backdated_bootstrap import (  # noqa: E402
    BackdatedSnapshot,
    BootstrapError,
    BootstrapResult,
    bootstrap_calibration,
    provenance_path,
    MODEL_VERSION_SUFFIX,
    GENERATOR_TAG,
)
from hot_theme_rotator.calibration.reporter import (  # noqa: E402
    build_calibration_report,
    derive_evidence_origin,
)
from hot_theme_rotator.calibration.schema import (  # noqa: E402
    ALLOWED_EVIDENCE_ORIGINS,
    CalibrationReport,
    CalibrationReportValidationError,
)
from hot_theme_rotator.decision_log.outcome_join import compute_outcomes  # noqa: E402
from hot_theme_rotator.decision_log.schema import PredictionRecord  # noqa: E402


# ─── stub historical loader & price fetcher ────────────────────────────────


class _StubLoader:
    def __init__(self, snapshots_by_date):
        self._snapshots = snapshots_by_date

    def load(self, *, trade_date: str):
        return self._snapshots.get(trade_date)


class _StubPriceFetcher:
    """Returns ~15 days of bars covering each horizon so outcome.status='complete'."""

    def __init__(self, base_price=100.0, drift=0.02):
        self._base = base_price
        self._drift = drift

    def fetch(self, *, symbol: str, start_date: str, end_date: str):
        from datetime import date, timedelta
        from hot_theme_rotator.common.schema import PriceBar
        d0 = date.fromisoformat(start_date)
        d1 = date.fromisoformat(end_date)
        bars = []
        price = self._base
        cur = d0
        while cur <= d1:
            price = price * (1.0 + self._drift)
            bars.append(PriceBar(
                symbol=symbol, asof=cur.isoformat(),
                open=price * 0.99, high=price * 1.01, low=price * 0.98, close=price,
                volume=1000.0, turnover_jpy=price * 1000.0,
            ))
            cur = cur + timedelta(days=1)
        return tuple(bars)


# ─── helpers ───────────────────────────────────────────────────────────────


def _snapshot(trade_date, candidates=None):
    if candidates is None:
        candidates = [{
            "symbol": "1306.T",
            "buy": 0.65,
            "sell": 0.0,
            "hold": 0.35,
            "score_status": "uncalibrated_research_score",
            "reference_price": 100.0,
        }]
    return BackdatedSnapshot(
        trade_date=trade_date,
        decision_cutoff=f"{trade_date}T15:00:00+09:00",
        input_snapshot_id=f"snap-{trade_date}",
        candidates=candidates,
    )


# ─── evidence_origin field ─────────────────────────────────────────────────


def test_calibration_report_accepts_each_evidence_origin():
    for origin in ALLOWED_EVIDENCE_ORIGINS:
        rep = CalibrationReport(
            source="opportunity", horizon_days=3,
            trade_date_range=("2026-05-01", "2026-05-25"),
            sample_count=0, status="insufficient_calibration",
            min_samples_required=100, evidence_origin=origin,
        )
        assert rep.evidence_origin == origin


def test_calibration_report_rejects_unknown_evidence_origin():
    with pytest.raises(CalibrationReportValidationError, match="evidence_origin"):
        CalibrationReport(
            source="opportunity", horizon_days=3,
            trade_date_range=("2026-05-01", "2026-05-25"),
            sample_count=0, status="insufficient_calibration",
            min_samples_required=100, evidence_origin="hearsay",
        )


def test_derive_evidence_origin_all_live():
    preds = [PredictionRecord.build(
        symbol="1306.T", trade_date="2026-05-25",
        decision_cutoff="2026-05-25T15:00:00+09:00",
        input_snapshot_id="s1", model_version="opportunity-v0",
        score_status="uncalibrated_research_score", horizon_days=3,
        buy=0.6, sell=0.0, hold=0.4, extra={"live": True},
    )]
    assert derive_evidence_origin(preds) == "live"


def test_derive_evidence_origin_all_bootstrap():
    preds = [PredictionRecord.build(
        symbol="1306.T", trade_date="2026-05-25",
        decision_cutoff="2026-05-25T15:00:00+09:00",
        input_snapshot_id="s1", model_version="opportunity-v0-backdated",
        score_status="uncalibrated_research_score", horizon_days=3,
        buy=0.6, sell=0.0, hold=0.4, extra={"backdated": True, "live": False},
    )]
    assert derive_evidence_origin(preds) == "bootstrap"


def test_derive_evidence_origin_mixed():
    live_pred = PredictionRecord.build(
        symbol="7203.T", trade_date="2026-05-25",
        decision_cutoff="2026-05-25T15:00:00+09:00",
        input_snapshot_id="s1", model_version="opportunity-v0",
        score_status="uncalibrated_research_score", horizon_days=3,
        buy=0.6, sell=0.0, hold=0.4,
    )
    bd_pred = PredictionRecord.build(
        symbol="1306.T", trade_date="2026-05-20",
        decision_cutoff="2026-05-20T15:00:00+09:00",
        input_snapshot_id="s2", model_version="opportunity-v0-backdated",
        score_status="uncalibrated_research_score", horizon_days=3,
        buy=0.7, sell=0.0, hold=0.3, extra={"backdated": True, "live": False},
    )
    assert derive_evidence_origin([live_pred, bd_pred]) == "mixed"


# ─── bootstrap orchestration ───────────────────────────────────────────────


def test_bootstrap_writes_predictions_with_backdated_flags(tmp_path):
    snaps = {f"2026-05-{d:02d}": _snapshot(f"2026-05-{d:02d}") for d in range(20, 25)}
    loader = _StubLoader(snaps)
    fetcher = _StubPriceFetcher()
    result = bootstrap_calibration(
        window_start="2026-05-20", window_end="2026-05-24",
        base_model_version="opportunity-v0",
        scanner_config_hash="sha:abc123", expected_scanner_config_hash="sha:abc123",
        snapshots_loader=loader, price_fetcher=fetcher,
        base_dir=tmp_path,
    )
    assert len(result.predictions) == 5
    for pred in result.predictions:
        assert pred.extra["backdated"] is True
        assert pred.extra["live"] is False
        assert pred.extra["generator"] == GENERATOR_TAG
        assert pred.model_version.endswith(MODEL_VERSION_SUFFIX)


def test_bootstrap_provenance_records_excluded_days(tmp_path):
    # Only 2 of 5 days have snapshots → 3 excluded
    snaps = {
        "2026-05-21": _snapshot("2026-05-21"),
        "2026-05-23": _snapshot("2026-05-23"),
    }
    loader = _StubLoader(snaps)
    fetcher = _StubPriceFetcher()
    result = bootstrap_calibration(
        window_start="2026-05-20", window_end="2026-05-24",
        base_model_version="opportunity-v0",
        scanner_config_hash="sha:abc123", expected_scanner_config_hash="sha:abc123",
        snapshots_loader=loader, price_fetcher=fetcher,
        base_dir=tmp_path,
    )
    excluded_dates = {e["trade_date"] for e in result.provenance.excluded}
    assert excluded_dates == {"2026-05-20", "2026-05-22", "2026-05-24"}
    assert all(e["reason"] == "no_archived_snapshot" for e in result.provenance.excluded)


def test_bootstrap_scanner_config_hash_mismatch_fail_closed(tmp_path):
    loader = _StubLoader({"2026-05-20": _snapshot("2026-05-20")})
    fetcher = _StubPriceFetcher()
    with pytest.raises(BootstrapError, match="scanner_config_hash"):
        bootstrap_calibration(
            window_start="2026-05-20", window_end="2026-05-20",
            base_model_version="opportunity-v0",
            scanner_config_hash="sha:wrong",
            expected_scanner_config_hash="sha:abc123",
            snapshots_loader=loader, price_fetcher=fetcher,
            base_dir=tmp_path,
        )
    assert not provenance_path(base_dir=tmp_path).exists()


def test_bootstrap_rejects_inverted_window(tmp_path):
    with pytest.raises(BootstrapError, match="window_start"):
        bootstrap_calibration(
            window_start="2026-05-25", window_end="2026-05-20",
            base_model_version="opportunity-v0",
            scanner_config_hash="x", expected_scanner_config_hash="x",
            snapshots_loader=_StubLoader({}), price_fetcher=_StubPriceFetcher(),
            base_dir=tmp_path,
        )


def test_bootstrap_rejects_cherry_picked_trading_days(tmp_path):
    """Providing an out-of-order trading_days list → BootstrapError."""
    with pytest.raises(BootstrapError, match="sorted"):
        bootstrap_calibration(
            window_start="2026-05-20", window_end="2026-05-24",
            base_model_version="opportunity-v0",
            scanner_config_hash="x", expected_scanner_config_hash="x",
            snapshots_loader=_StubLoader({}), price_fetcher=_StubPriceFetcher(),
            base_dir=tmp_path,
            trading_days=["2026-05-24", "2026-05-20"],  # out of order
        )


def test_bootstrap_rejects_trading_day_outside_window(tmp_path):
    with pytest.raises(BootstrapError, match="extend beyond"):
        bootstrap_calibration(
            window_start="2026-05-20", window_end="2026-05-22",
            base_model_version="opportunity-v0",
            scanner_config_hash="x", expected_scanner_config_hash="x",
            snapshots_loader=_StubLoader({}), price_fetcher=_StubPriceFetcher(),
            base_dir=tmp_path,
            trading_days=["2026-05-20", "2026-05-25"],
        )


def test_bootstrap_provenance_json_payload(tmp_path):
    snaps = {"2026-05-21": _snapshot("2026-05-21")}
    loader = _StubLoader(snaps)
    fetcher = _StubPriceFetcher()
    bootstrap_calibration(
        window_start="2026-05-20", window_end="2026-05-22",
        base_model_version="opportunity-v0",
        scanner_config_hash="sha:abc123", expected_scanner_config_hash="sha:abc123",
        snapshots_loader=loader, price_fetcher=fetcher,
        base_dir=tmp_path,
    )
    path = provenance_path(base_dir=tmp_path)
    assert path.exists()
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["window_start"] == "2026-05-20"
    assert payload["window_end"] == "2026-05-22"
    assert payload["model_version"] == "opportunity-v0-backdated"
    assert payload["scanner_config_hash"] == "sha:abc123"
    assert payload["snapshots_loaded"] == 1
    assert payload["generator"] == GENERATOR_TAG
    assert len(payload["excluded"]) == 2


def test_bootstrap_outcomes_complete_against_stub_fetcher(tmp_path):
    snaps = {f"2026-05-{d:02d}": _snapshot(f"2026-05-{d:02d}") for d in range(20, 22)}
    loader = _StubLoader(snaps)
    fetcher = _StubPriceFetcher()
    result = bootstrap_calibration(
        window_start="2026-05-20", window_end="2026-05-21",
        base_model_version="opportunity-v0",
        scanner_config_hash="x", expected_scanner_config_hash="x",
        snapshots_loader=loader, price_fetcher=fetcher,
        base_dir=tmp_path,
    )
    assert result.outcomes_built == 2
    assert result.outcomes_complete == 2  # stub fetcher gives full horizon coverage


def test_bootstrap_calibration_report_marks_evidence_origin_bootstrap(tmp_path):
    snaps = {f"2026-05-{d:02d}": _snapshot(f"2026-05-{d:02d}") for d in range(1, 26)}
    loader = _StubLoader(snaps)
    fetcher = _StubPriceFetcher()
    result = bootstrap_calibration(
        window_start="2026-05-01", window_end="2026-05-25",
        base_model_version="opportunity-v0",
        scanner_config_hash="x", expected_scanner_config_hash="x",
        snapshots_loader=loader, price_fetcher=fetcher,
        base_dir=tmp_path,
    )
    fetcher2 = _StubPriceFetcher()
    summary = compute_outcomes(
        result.predictions, fetcher=fetcher2,
        evaluated_as_of="2026-06-01",
    )
    # min_samples lower for test sample size
    report = build_calibration_report(
        predictions=result.predictions, outcomes=summary.outcomes,
        source="opportunity", horizon_days=3, min_samples=10,
    )
    assert report.evidence_origin == "bootstrap"
    assert report.status == "calibrated"
