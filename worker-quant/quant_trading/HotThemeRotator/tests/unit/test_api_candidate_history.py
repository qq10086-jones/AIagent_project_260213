"""Tests for GET /api/candidates/history — cohort review endpoint (Rule 11.11).

These run against the real on-disk decision log (reports/predictions +
reports/outcomes), mirroring the dashboard tests' fail-soft style: shape and
honesty invariants are asserted, not data-dependent numerics.

P37-05 amendment: three cases here DID assert data-dependent facts - that
2026-03-23 and 2026-05-27 are present in the log - which is exactly what the
docstring said they would not do. Both directories are gitignored runtime
state, so those cases passed only on a machine that had already run the
pipeline, and failed in a clean git worktree and would have failed in CI. The
structural assertions still run everywhere; the on-disk-truth ones now skip by
name when the log is absent, rather than failing or being quietly deleted.
"""
import sys
import json
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi.testclient import TestClient  # noqa: E402

from api.main import create_app  # noqa: E402


@pytest.fixture
def client():
    return TestClient(create_app())


_LOG_DATES = ("2026-03-23", "2026-05-27")


def _require_decision_log(client):
    """Skip, by name, when the gitignored decision log is not in this checkout."""
    payload = client.get("/api/candidates/history/dates").json()
    present = set(payload.get("live") or []) | set(payload.get("backdated") or [])
    missing = [d for d in _LOG_DATES if d not in present]
    if missing:
        pytest.skip(
            "the on-disk decision log (reports/predictions + reports/outcomes) is "
            f"gitignored runtime state and this checkout lacks {missing}; the "
            "structural assertions in this module still run"
        )


def test_dates_endpoint_buckets_live_and_backdated(client):
    _require_decision_log(client)
    payload = client.get("/api/candidates/history/dates").json()
    assert "live" in payload and "backdated" in payload
    assert isinstance(payload["live"], list)
    assert isinstance(payload["backdated"], list)
    # On-disk truth: 2026-03-23 is bootstrap (backdated), 2026-05-27 is live.
    assert "2026-03-23" in payload["backdated"]
    assert "2026-05-27" in payload["live"]
    # A date is never in both buckets.
    assert not (set(payload["live"]) & set(payload["backdated"]))


def test_history_for_live_date_has_required_shape(client):
    payload = client.get("/api/candidates/history?date=2026-05-27").json()
    for key in ("tradeDate", "benchmarkSymbol", "candidateCount", "excludedBackdated",
                "completeCount", "sampleSufficient", "minSamples", "honestyNote",
                "horizons", "candidates", "availableDates"):
        assert key in payload, f"missing key: {key}"
    assert payload["tradeDate"] == "2026-05-27"
    assert payload["benchmarkSymbol"] == "1306.T"
    # Three forward horizons, cohort-first aggregate shape (Rule 11.11.2).
    assert [h["horizon"] for h in payload["horizons"]] == ["1D", "3D", "5D"]
    for h in payload["horizons"]:
        for key in ("maturedCount", "immatureCount", "meanReturn",
                    "positiveShare", "benchmarkReturn", "excessReturn"):
            assert key in h
    for c in payload["candidates"]:
        for key in ("symbol", "buyScore", "referencePrice", "decisionCutoff",
                    "scoreStatus", "outcomeStatus", "realizedReturns"):
            assert key in c
        assert c["scoreStatus"] == "uncalibrated_research_score"


def test_default_date_is_latest_live(client):
    dates = client.get("/api/candidates/history/dates").json()
    payload = client.get("/api/candidates/history").json()
    if dates["live"]:
        assert payload["tradeDate"] == dates["live"][-1]


def test_backdated_date_default_excluded_shows_empty_roster(client):
    _require_decision_log(client)
    """Rule 11.11.5 — a bootstrap date renders no live roster by default."""
    payload = client.get("/api/candidates/history?date=2026-03-23").json()
    assert payload["candidateCount"] == 0
    assert payload["candidates"] == []
    assert payload["excludedBackdated"] > 0


def test_backdated_date_surfaces_when_explicitly_included(client):
    _require_decision_log(client)
    payload = client.get(
        "/api/candidates/history?date=2026-03-23&include_backdated=true"
    ).json()
    assert payload["candidateCount"] > 0
    assert payload["excludedBackdated"] == 0


def test_payload_carries_no_win_rate_or_probability_language(client):
    """Rule 11.11.6 / 8.3 / 9.4 — cohort stats are observations, not win rates."""
    rendered = json.dumps(
        client.get("/api/candidates/history?date=2026-05-27").json(),
        ensure_ascii=False,
    ).lower()
    assert "win rate" not in rendered
    assert "probability" not in rendered
    # No field labels the score as calibrated / a probability.
    assert "calibrated_probability" not in rendered
    assert "winrate" not in rendered
    # The standing honesty disclosure must be present (it may itself use the word
    # 胜率 in the negative — "样本不足以得出胜率结论" — exactly like the dashboard's
    # "不是真实胜率"; that is a disclaimer, not a claim).
    assert "未校准研究分数" in rendered
    assert "样本不足以得出胜率结论" in rendered


def test_history_rejects_post_method(client):
    """Rule 3 / Section 11 — read-only surface, no write path."""
    assert client.post("/api/candidates/history").status_code == 405
    assert client.post("/api/candidates/history/dates").status_code == 405


def test_history_immature_samples_excluded_from_aggregate_but_counted(client):
    """Survivorship guard (Rule 11.11.3): a candidate without a complete outcome
    inflates immatureCount, never silently vanishes."""
    payload = client.get("/api/candidates/history?date=2026-05-27").json()
    cohort = payload["candidateCount"]
    for h in payload["horizons"]:
        assert h["maturedCount"] + h["immatureCount"] == cohort


# ── cross-day trend endpoint (Rule 11.11.2 full-series view) ────────────────

def test_trend_endpoint_shape(client):
    payload = client.get("/api/candidates/history/trend").json()
    for key in ("benchmarkSymbol", "totalComplete", "sampleSufficient",
                "minSamples", "honestyNote", "pooled", "points"):
        assert key in payload, f"missing key: {key}"
    assert payload["benchmarkSymbol"] == "1306.T"
    assert [p["horizon"] for p in payload["pooled"]] == ["1D", "3D", "5D"]
    for p in payload["pooled"]:
        for key in ("maturedCount", "meanReturn", "positiveShare",
                    "benchmarkReturn", "excessReturn"):
            assert key in p


def test_trend_points_one_per_live_date(client):
    dates = client.get("/api/candidates/history/dates").json()["live"]
    payload = client.get("/api/candidates/history/trend").json()
    assert [pt["tradeDate"] for pt in payload["points"]] == dates
    for pt in payload["points"]:
        for key in ("tradeDate", "candidateCount", "completeCount", "horizons"):
            assert key in pt


def test_trend_total_complete_is_sum_of_points(client):
    payload = client.get("/api/candidates/history/trend").json()
    assert payload["totalComplete"] == sum(pt["completeCount"] for pt in payload["points"])
    assert payload["sampleSufficient"] is (payload["totalComplete"] >= payload["minSamples"])


def test_trend_rejects_post_and_carries_disclosure(client):
    assert client.post("/api/candidates/history/trend").status_code == 405
    rendered = json.dumps(client.get("/api/candidates/history/trend").json(),
                          ensure_ascii=False).lower()
    assert "win rate" not in rendered
    assert "probability" not in rendered
    assert "未校准研究分数" in rendered
