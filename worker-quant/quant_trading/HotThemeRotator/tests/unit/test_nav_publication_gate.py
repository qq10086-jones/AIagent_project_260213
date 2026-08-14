"""P37-04 — an unreconciled NAV must not enter the official history.

The 2026-08-14 artifact is the specimen: `reports/observability/risk_mandate/
2026-08-14.json` recorded NAV JPY 394,724 as ordinary history while the broker
account totalled JPY 393,998. Nothing marked it. The trace is what later
analyses read, and a bare NAV there is indistinguishable from a settled one.

These tests pin the gate rather than the arithmetic: whatever the numbers,
a snapshot written without external reconciliation must say so in the artifact,
and must not be counted as an official, metric-bearing NAV.
"""
from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import pytest  # noqa: E402

from hot_theme_rotator.portfolio.broker_reconciliation import (  # noqa: E402
    MISMATCH,
    MISSING_BROKER_SNAPSHOT,
    RECONCILED,
    BrokerPosition,
    BrokerSnapshot,
    JournalView,
    reconcile_against_broker,
)
from hot_theme_rotator.portfolio.nav_publication import (  # noqa: E402
    NAV_STATUS_KEY,
    UNRECONCILED_NAV_NOTE,
    annotate_nav_record,
    may_publish_official_nav,
    supersede_record,
)


def _broker() -> BrokerSnapshot:
    return BrokerSnapshot(
        asof=date(2026, 8, 14),
        cash=287_068.0,
        positions={
            "1306.T": BrokerPosition("1306.T", 100, 437.8, 43_780.0),
            "1568.T": BrokerPosition("1568.T", 60, 1052.5, 63_150.0),
        },
        total_assets=393_998.0,
        source="sbi_web_manual",
        mark_time="2026-08-14T15:30+09:00",
    )


def _view(cash: float) -> JournalView:
    return JournalView(
        asof=date(2026, 8, 14),
        cash=cash,
        quantities={"1306.T": 100, "1568.T": 60},
        mark_time="2026-08-14T15:30+09:00",
    )


# ---------------------------------------------------------------------------
# The gate
# ---------------------------------------------------------------------------
def test_no_broker_snapshot_means_no_official_nav():
    verdict = reconcile_against_broker(_view(287_068.0), None)
    assert verdict.state == MISSING_BROKER_SNAPSHOT
    assert may_publish_official_nav(verdict) is False


def test_a_mismatch_means_no_official_nav():
    verdict = reconcile_against_broker(_view(287_794.0), _broker())
    assert verdict.state == MISMATCH
    assert may_publish_official_nav(verdict) is False


def test_agreement_unlocks_official_nav():
    verdict = reconcile_against_broker(_view(287_068.0), _broker())
    assert verdict.state == RECONCILED
    assert may_publish_official_nav(verdict) is True


# ---------------------------------------------------------------------------
# The artifact must carry its own status
# ---------------------------------------------------------------------------
def test_an_unreconciled_record_is_labelled_in_the_artifact():
    verdict = reconcile_against_broker(_view(287_794.0), _broker())
    record = annotate_nav_record({"asof": "2026-08-14", "nav_jpy": 394_724.0}, verdict)
    assert record[NAV_STATUS_KEY]["state"] == MISMATCH
    assert record[NAV_STATUS_KEY]["official"] is False
    assert record[NAV_STATUS_KEY]["metrics_allowed"] is False
    assert UNRECONCILED_NAV_NOTE in record[NAV_STATUS_KEY]["note"]
    # The number is kept - deleting evidence is not the fix.
    assert record["nav_jpy"] == 394_724.0


def test_a_reconciled_record_is_labelled_official():
    verdict = reconcile_against_broker(_view(287_068.0), _broker())
    record = annotate_nav_record({"asof": "2026-08-14", "nav_jpy": 393_998.0}, verdict)
    assert record[NAV_STATUS_KEY]["official"] is True
    assert record[NAV_STATUS_KEY]["metrics_allowed"] is True


def test_annotation_never_edits_the_nav_it_labels():
    """Labelling is not correcting; the figure must survive untouched."""
    verdict = reconcile_against_broker(_view(287_794.0), _broker())
    original = {"asof": "2026-08-14", "nav_jpy": 394_724.0, "exposure_ratio": 0.432}
    record = annotate_nav_record(dict(original), verdict)
    for key, value in original.items():
        assert record[key] == value


def test_annotation_carries_the_differences_but_proposes_nothing():
    verdict = reconcile_against_broker(_view(287_794.0), _broker())
    record = annotate_nav_record({"asof": "2026-08-14", "nav_jpy": 394_724.0}, verdict)
    status = record[NAV_STATUS_KEY]
    assert any(d["field"] == "cash" for d in status["differences"])
    assert "proposed_correction" not in status
    assert "adjustment" not in json.dumps(status).lower()


# ---------------------------------------------------------------------------
# Superseding, append-only in spirit
# ---------------------------------------------------------------------------
def test_superseding_keeps_the_original_and_names_the_reason():
    record = {"asof": "2026-08-14", "nav_jpy": 394_724.0}
    superseded = supersede_record(record, reason="broker cash disagrees by JPY 726")
    assert superseded["superseded"] is True
    assert "726" in superseded["superseded_reason"]
    assert superseded["nav_jpy"] == 394_724.0, "the wrong figure stays visible"


def test_a_superseded_record_is_never_official():
    verdict = reconcile_against_broker(_view(287_794.0), _broker())
    record = supersede_record(
        annotate_nav_record({"asof": "2026-08-14", "nav_jpy": 394_724.0}, verdict),
        reason="unreconciled against SBI 2026-08-14",
    )
    assert record[NAV_STATUS_KEY]["official"] is False
    assert record["superseded"] is True


# ---------------------------------------------------------------------------
# The live artifact on disk
# ---------------------------------------------------------------------------
ARTIFACT = PROJECT_ROOT / "reports" / "observability" / "risk_mandate" / "2026-08-14.json"


@pytest.mark.skipif(not ARTIFACT.is_file(), reason="the 2026-08-14 artifact is gitignored runtime state")
def test_the_known_unreconciled_artifact_is_labelled_on_disk():
    """The specimen itself must not sit in history looking settled."""
    payload = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    status = payload.get(NAV_STATUS_KEY)
    assert status is not None, (
        "the 2026-08-14 risk-mandate artifact carries a NAV that disagrees with "
        "the broker and must be labelled unreconciled"
    )
    assert status["official"] is False
