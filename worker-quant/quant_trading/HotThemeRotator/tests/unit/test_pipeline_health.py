"""Tests for the pipeline health state contract (P37-01, Rule 15.10).

The defect being pinned: `ok` meant "core collection succeeded" and was read as
"the pipeline is healthy". TDnet polling exited non-zero on five real afterclose
sessions with `ok: true` every time, and TDnet serves its documents for only
~31 days — so a silent degraded day there is permanent data loss.

Every invariant below serves the separation. `ok` keeps its old meaning and is
never widened. `health_status == "failed"` iff `ok is False`, so degraded can
neither masquerade as failure nor hide inside healthy. Components carry stable
codes, because a degradation that cannot be named cannot be counted. And a
component that produced no result is `not_run` — degraded — never dropped from
the roster and never scored as success.
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import pytest  # noqa: E402

from hot_theme_rotator.observability.pipeline_health import (  # noqa: E402
    AFTERCLOSE_COMPONENTS,
    HEALTH_DEGRADED,
    HEALTH_FAILED,
    HEALTH_HEALTHY,
    STATUS_NOT_RUN,
    STATUS_OK,
    STATUS_PARTIAL,
    STATUS_SKIPPED,
    assess_record,
    exit_code_for,
)


def _healthy_record(**overrides) -> dict:
    """A fully green afterclose record, shaped exactly like the real log."""
    record = {
        "ts": "2026-08-11T19:30:01+09:00",
        "mode": "afterclose",
        "asof": "2026-08-10",
        "dry_run": False,
        "ok": True,
        "candidates": {
            "ok": True,
            "ticker_count": 50,
            "htr_db_refresh_rc": 0,
            "event_maintenance": "ok",
            "news_refresh_rc": 0,
            "macro_refresh_rc": 0,
            "meta_refresh_rc": 0,
            "s_kabu_overlay_rc": 0,
            "adr_refresh_rc": 0,
            "tdnet_poll_rc": 0,
            "revision_capture_rc": 0,
        },
        # Real shape: collect() returns raw return codes and counts, NOT an
        # `ok` key. The combined verdict lives in the record's top-level `ok`.
        "collection": {"emit_rc": 0, "sweep_rc": 0, "new_predictions": 50,
                       "dropped_no_close": 0, "skipped_on_disk": 0},
        "forward_eval": {"rc": 0, "tail": ""},
        "cohort": {"emit_rc": "skipped_month_exists", "sweep_rc": 0},
        "value_livelog": {"rc": 0, "tail": ""},
        "risk_mandate": {"rc": 0, "tail": ""},
    }
    record.update(overrides)
    return record


def _codes(health: dict) -> set[str]:
    return {row["code"] for row in health["degraded_components"]}


# ── 1. healthy ───────────────────────────────────────────────────────────


def test_a_fully_green_run_is_healthy():
    health = assess_record(_healthy_record())

    assert health["health_status"] == HEALTH_HEALTHY
    assert health["degraded_components"] == []
    assert health["perishable_degraded"] == []
    assert exit_code_for(health["health_status"]) == 0
    # Every declared component appears in the roster, healthy or not.
    assert {c["component"] for c in health["components"]} == {
        s.name for s in AFTERCLOSE_COMPONENTS}


def test_the_monthly_cohort_skip_is_a_skip_not_a_degradation():
    health = assess_record(_healthy_record())
    cohort = next(c for c in health["components"] if c["component"] == "cohort")
    # Cadence, not a missing result. Collapsing the two would cry wolf monthly
    # or hide a real gap eleven times out of twelve.
    assert cohort["status"] == STATUS_SKIPPED
    assert cohort["code"] == "cohort.monthly_emit_already_done"
    assert health["health_status"] == HEALTH_HEALTHY


# ── 2. TDnet non-zero: the five real sessions ────────────────────────────


def test_tdnet_nonzero_degrades_while_ok_stays_true():
    record = _healthy_record()
    record["candidates"]["tdnet_poll_rc"] = 1

    health = assess_record(record)

    assert record["ok"] is True          # unchanged: core collection did succeed
    assert health["health_status"] == HEALTH_DEGRADED
    assert _codes(health) == {"tdnet_poll.nonzero_exit"}
    assert exit_code_for(health["health_status"]) == 3


def test_tdnet_degradation_is_labelled_perishable():
    record = _healthy_record()
    record["candidates"]["tdnet_poll_rc"] = 1

    health = assess_record(record)

    # ~31-day TDnet retention: "we'll catch it next run" is false here, and the
    # health report has to carry that distinction.
    assert health["perishable_degraded"] == ["tdnet_poll"]
    row = next(r for r in health["degraded_components"]
               if r["component"] == "tdnet_poll")
    assert row["perishable"] is True
    assert row["detail"] == "rc=1"


def test_a_non_perishable_degradation_is_not_labelled_perishable():
    record = _healthy_record()
    record["candidates"]["news_refresh_rc"] = 1

    health = assess_record(record)

    assert health["health_status"] == HEALTH_DEGRADED
    assert health["perishable_degraded"] == []


# ── 3. event-universe partial ────────────────────────────────────────────


def test_event_universe_partial_degrades_without_failing():
    record = _healthy_record()
    record["candidates"]["event_maintenance"] = "event_universe_partial"
    record["candidates"]["htr_db_refresh_rc"] = 3

    health = assess_record(record)

    assert health["health_status"] == HEALTH_DEGRADED
    assert _codes(health) == {"price_db_refresh.event_universe_partial"}
    row = next(r for r in health["degraded_components"]
               if r["component"] == "price_db_refresh")
    assert row["status"] == STATUS_PARTIAL


def test_the_real_2026_08_10_record_is_degraded_not_ok():
    """The exact shape that logged `ok: true` with two live degradations."""
    record = _healthy_record()
    record["candidates"]["event_maintenance"] = "event_universe_partial"
    record["candidates"]["htr_db_refresh_rc"] = 3
    record["candidates"]["tdnet_poll_rc"] = 1

    health = assess_record(record)

    assert health["health_status"] == HEALTH_DEGRADED
    assert _codes(health) == {
        "price_db_refresh.event_universe_partial", "tdnet_poll.nonzero_exit"}
    assert health["perishable_degraded"] == ["tdnet_poll"]


def test_an_unrecognised_maintenance_state_fails_closed():
    record = _healthy_record()
    record["candidates"]["event_maintenance"] = "something_new"

    health = assess_record(record)

    assert _codes(health) == {"price_db_refresh.unknown_maintenance_state"}


# ── 4. multiple simultaneous degradations ────────────────────────────────


def test_every_simultaneous_degradation_is_named():
    record = _healthy_record()
    record["candidates"]["tdnet_poll_rc"] = 1
    record["candidates"]["revision_capture_rc"] = 2
    record["candidates"]["event_maintenance"] = "event_universe_partial"
    record["forward_eval"] = {"rc": None, "error": "TimeoutExpired"}
    record["risk_mandate"] = {"rc": 1, "tail": ""}

    health = assess_record(record)

    assert health["health_status"] == HEALTH_DEGRADED
    assert _codes(health) == {
        "tdnet_poll.nonzero_exit",
        "revision_capture.nonzero_exit",
        "price_db_refresh.event_universe_partial",
        "forward_eval.exception",
        "risk_mandate.nonzero_exit",
    }
    # Rule 15.10.7: the aggregate is never publishable without its components.
    assert len(health["degraded_components"]) == 5
    assert sorted(health["perishable_degraded"]) == ["revision_capture", "tdnet_poll"]
    assert health["summary"].startswith("degraded: ")


def test_summary_names_the_codes_not_just_the_count():
    record = _healthy_record()
    record["candidates"]["tdnet_poll_rc"] = 1

    assert assess_record(record)["summary"] == "degraded: tdnet_poll.nonzero_exit"


# ── 5. core failure ──────────────────────────────────────────────────────


def test_a_failed_candidate_refresh_is_failed_not_degraded():
    record = _healthy_record(ok=False)
    record["candidates"] = {"ok": False, "reason": "screener produced 0 candidates"}

    health = assess_record(record)

    assert health["health_status"] == HEALTH_FAILED
    assert exit_code_for(health["health_status"]) == 1
    codes = _codes(health)
    assert "candidate_refresh.failed" in codes
    row = next(r for r in health["degraded_components"]
               if r["component"] == "candidate_refresh")
    assert row["detail"] == "screener produced 0 candidates"
    assert row["core"] is True


def test_the_zero_new_sample_guard_is_a_core_failure():
    """Green return codes with zero new samples and nothing skipped."""
    record = _healthy_record(ok=False)
    record["collection"] = {"emit_rc": 0, "sweep_rc": 0, "new_predictions": 0,
                            "dropped_no_close": 12, "skipped_on_disk": 0}

    health = assess_record(record)

    assert health["health_status"] == HEALTH_FAILED
    assert "forward_collection.no_new_samples" in _codes(health)


def test_a_nonzero_emit_or_sweep_is_a_core_failure():
    record = _healthy_record(ok=False)
    record["collection"] = {"emit_rc": 2, "sweep_rc": 0, "new_predictions": None}

    health = assess_record(record)

    assert health["health_status"] == HEALTH_FAILED
    row = next(r for r in health["degraded_components"]
               if r["component"] == "forward_collection")
    assert row["code"] == "forward_collection.nonzero_exit"
    assert "emit_rc=2" in row["detail"]


def test_ok_false_is_always_failed_and_the_cause_is_attributed():
    """The invariant binds in both directions (Rule 15.10.2)."""
    health = assess_record(_healthy_record(ok=False))

    assert health["health_status"] == HEALTH_FAILED
    # `ok=false` on an otherwise-green afterclose record can only mean the
    # zero-new-sample guard fired, and the roster says so rather than shrugging.
    assert "forward_collection.no_new_samples" in _codes(health)


def test_an_unattributable_core_failure_is_named_not_softened():
    """A contradictory record — `ok=false` while every core component reports
    healthy — means the roster no longer covers everything `ok` covers (a new
    core gate added without a component). That is reported as an unexplained
    core failure, never downgraded to `degraded`."""
    health = assess_record({
        "mode": "preopen", "ok": False,
        "smoke": {"passed": True, "rc": 0, "summary": "2358 passed"},
        "prior_session_snapshot_present": True})

    assert health["health_status"] == HEALTH_FAILED
    assert "core.unspecified" in _codes(health)


def test_a_core_failure_does_not_suppress_the_other_degradations():
    record = _healthy_record(ok=False)
    record["candidates"] = {"ok": False, "reason": "screener exit 2",
                            "tdnet_poll_rc": 1}

    health = assess_record(record)

    assert health["health_status"] == HEALTH_FAILED
    assert {"candidate_refresh.failed", "tdnet_poll.nonzero_exit"} <= _codes(health)


# ── 6. a component that did not run at all ───────────────────────────────


def test_a_missing_component_is_not_run_and_degrades():
    """Silence is not health (Rule 15.10.4)."""
    record = _healthy_record()
    del record["candidates"]["tdnet_poll_rc"]
    del record["value_livelog"]

    health = assess_record(record)

    assert health["health_status"] == HEALTH_DEGRADED
    assert _codes(health) == {"tdnet_poll.not_run", "value_livelog.not_run"}
    # Still on the roster — a component cannot vanish by failing to report.
    names = {c["component"] for c in health["components"]}
    assert {"tdnet_poll", "value_livelog"} <= names
    for row in health["degraded_components"]:
        assert row["status"] == STATUS_NOT_RUN


def test_an_early_return_that_skips_maintenance_is_reported_not_assumed():
    """The screener-timeout path returns without most rc fields."""
    record = _healthy_record(ok=False)
    record["candidates"] = {"ok": False, "reason": "screener timeout (300s)",
                            "htr_db_refresh_rc": 0, "event_maintenance": "ok"}

    health = assess_record(record)

    assert health["health_status"] == HEALTH_FAILED
    # Everything the early return skipped is not_run, not silently ok.
    assert "tdnet_poll.not_run" in _codes(health)
    assert "news_refresh.not_run" in _codes(health)


# ── 7. idempotent re-run ─────────────────────────────────────────────────


def test_an_idempotent_rerun_with_no_new_samples_is_still_healthy():
    """`ok` covers this case already; health must not contradict it.

    Regression: the first version of this component read a non-existent
    ``collection["ok"]`` key, so every idempotent re-run — the normal state of
    a re-run day — was scored a CORE FAILURE. The unit tests missed it because
    the fixture invented an `ok` key the producer never writes; replaying a
    real log record is what caught it.
    """
    record = _healthy_record()
    record["collection"] = {"emit_rc": 0, "sweep_rc": 0, "new_predictions": 0,
                            "dropped_no_close": 0, "skipped_on_disk": 50}
    record["notes"] = ["already collected for 2026-08-10 (idempotent, skipped=50)"]

    health = assess_record(record)

    assert health["health_status"] == HEALTH_HEALTHY


def test_the_real_logged_record_shape_is_assessed_correctly():
    """Replay of the actual 2026-08-10 afterclose row from the production log.

    This is the shape the producer really writes — no invented keys — and it
    must come out DEGRADED (two live degradations) rather than failed.
    """
    record = {
        "ts": "2026-08-11T19:30:01+09:00", "mode": "afterclose",
        "asof": "2026-08-10", "dry_run": False, "calendar_covered": True,
        "notes": ["already collected for 2026-08-10 (idempotent, skipped=50); "
                  "no new samples"],
        "candidates": {
            "ok": True, "out": "…selected_tickers_2026-08-10.json",
            "ticker_count": 50, "trade_date": "2026-08-10",
            "htr_db_refresh_rc": 3, "event_maintenance": "event_universe_partial",
            "news_refresh_rc": 0, "macro_refresh_rc": 0, "meta_refresh_rc": 0,
            "s_kabu_overlay_rc": 0, "adr_refresh_rc": 0,
            "tdnet_poll_rc": 1, "revision_capture_rc": 0,
        },
        "collection": {"emit_rc": 0, "emit_tail": "", "sweep_rc": 0,
                       "sweep_tail": "", "new_predictions": 0,
                       "dropped_no_close": 0, "skipped_on_disk": 50},
        "forward_eval": {"rc": 0, "tail": ""},
        "cohort": {"emit_rc": "skipped_month_exists", "sweep_rc": 0},
        "value_livelog": {"rc": 0, "tail": ""},
        "risk_mandate": {"rc": 0, "tail": ""},
        "ok": True,
    }

    health = assess_record(record)

    assert health["health_status"] == HEALTH_DEGRADED
    assert _codes(health) == {
        "price_db_refresh.event_universe_partial", "tdnet_poll.nonzero_exit"}
    assert health["perishable_degraded"] == ["tdnet_poll"]
    assert next(c for c in health["components"]
                if c["component"] == "forward_collection")["status"] == STATUS_OK


def test_assessment_is_pure_and_repeatable():
    record = _healthy_record()
    record["candidates"]["tdnet_poll_rc"] = 1
    before = repr(record)

    first = assess_record(record)
    second = assess_record(record)

    assert first == second
    assert repr(record) == before  # the reporter never mutates what it reports


# ── preopen + dry-run ────────────────────────────────────────────────────


def test_preopen_uses_its_own_component_roster():
    record = {"mode": "preopen", "asof": "2026-08-10", "ok": True,
              "smoke": {"passed": True, "rc": 0, "summary": "2358 passed"},
              "candidate_snapshot_present": False,
              "prior_session_asof": "2026-08-07",
              "prior_session_snapshot_present": True}

    health = assess_record(record)

    assert health["health_status"] == HEALTH_HEALTHY
    assert {c["component"] for c in health["components"]} == {
        "smoke", "candidate_snapshot"}
    assert next(c for c in health["components"]
                if c["component"] == "smoke")["detail"] == "2358 passed"


def test_preopen_missing_snapshot_degrades_but_a_failed_smoke_fails():
    degraded = assess_record({
        "mode": "preopen", "ok": True,
        "smoke": {"passed": True, "rc": 0, "summary": "ok"},
        "prior_session_snapshot_present": False})
    assert degraded["health_status"] == HEALTH_DEGRADED
    assert _codes(degraded) == {"candidate_snapshot.absent"}

    failed = assess_record({
        "mode": "preopen", "ok": False,
        "smoke": {"passed": False, "rc": 1, "summary": "3 failed"},
        "prior_session_snapshot_present": True})
    assert failed["health_status"] == HEALTH_FAILED
    assert "smoke.smoke_failed" in _codes(failed)


def test_a_dry_run_is_not_healthy_because_it_measured_nothing():
    health = assess_record({"mode": "afterclose", "dry_run": True, "ok": True,
                            "plan": ["..."]})

    assert health["health_status"] == HEALTH_DEGRADED
    assert _codes(health) == {"dry_run.no_collection"}


@pytest.mark.parametrize("status,code", [
    (HEALTH_HEALTHY, 0), (HEALTH_DEGRADED, 3), (HEALTH_FAILED, 1),
    ("anything_unknown", 1),
])
def test_exit_codes_follow_the_declared_mapping(status, code):
    # Unknown states fail closed at 1 rather than reporting success.
    assert exit_code_for(status) == code


def test_component_codes_are_unique_and_stable():
    """A code is an identity that will be counted over months."""
    names = [s.name for s in AFTERCLOSE_COMPONENTS]
    assert len(names) == len(set(names))
    health = assess_record(_healthy_record())
    for row in health["components"]:
        assert row["code"].startswith(row["component"] + ".")
        assert row["status"] in {STATUS_OK, STATUS_SKIPPED}


def test_a_genuine_0830_preopen_is_healthy_not_perpetually_degraded():
    """Regression: the pre-open snapshot check was structurally always false.

    `candidate_snapshot_present` asks whether ASOF's snapshot is on disk, and at
    08:30 on a trading day asof IS today — whose snapshot is written at that
    day's afterclose. Real logs: false on 7 of 7 genuine pre-open runs. Wiring
    that into health would have emitted `degraded` every single morning, which
    teaches the operator to ignore the aggregate entirely. Health reads the
    question that has an answer at pre-open: is the LAST COMPLETED session's
    snapshot ready.
    """
    health = assess_record({
        "mode": "preopen", "asof": "2026-08-10", "ok": True,
        "smoke": {"passed": True, "rc": 0, "summary": "2358 passed"},
        "candidate_snapshot_present": False,      # structurally false at 08:30
        "prior_session_asof": "2026-08-07",
        "prior_session_snapshot_present": True,   # yesterday's afterclose ran
    })

    assert health["health_status"] == HEALTH_HEALTHY
    assert health["degraded_components"] == []
