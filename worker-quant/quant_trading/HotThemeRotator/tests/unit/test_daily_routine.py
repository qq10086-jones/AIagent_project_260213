"""Tests for the Local Beta v0 daily routine orchestrator (Rule 15.5).

The orchestrator must be deterministic, fail-closed, advice-only, and never
invoke an execution or LLM path. All subprocess work is injected so these tests
stay in the fast daily smoke lane.
"""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import tools.daily_routine as dr


def test_latest_trading_day_steps_weekend_back_to_friday():
    assert dr.latest_trading_day(date(2026, 5, 30)) == date(2026, 5, 29)  # Sat -> Fri
    assert dr.latest_trading_day(date(2026, 5, 31)) == date(2026, 5, 29)  # Sun -> Fri
    assert dr.latest_trading_day(date(2026, 5, 27)) == date(2026, 5, 27)  # Wed -> Wed


def test_latest_trading_day_steps_over_jpx_holiday(tmp_path):
    """P12-03(b) — the routine's trade date now steps back over JP holidays, not
    just weekends, so emit never targets a closed session (Rule 15.4)."""
    # Golden Week 2026: 5/6 Wed (substitute holiday) -> back through 5/4/5/5 + the
    # 5/2-5/3 weekend to Friday 5/1.
    assert dr.latest_trading_day(date(2026, 5, 6)) == date(2026, 5, 1)
    assert dr.latest_trading_day(date(2026, 4, 29)) == date(2026, 4, 28)  # Showa Wed -> Tue


def test_afterclose_record_flags_calendar_coverage(tmp_path, monkeypatch):
    """Rule 15.4 honesty — the run record states whether the JPX holiday table
    covers asof's year (out-of-coverage falls back to weekend-only + data guard)."""
    monkeypatch.setattr(dr, "SNAP_DIR", tmp_path / "screener")
    runner, _ = _fake_runner_factory(snapshot_payload={"asof": "2026-05-29", "symbols": ["6966.T"]})
    rec = dr.run_afterclose(date(2026, 5, 30), runner=runner)
    assert rec["calendar_covered"] is True


def _fake_runner_factory(snapshot_payload=None, screener_rc=0, calls=None,
                         emit_new=2, emit_dropped=0):
    """Build a runner that records calls and (optionally) writes a snapshot.

    emit returns realistic count lines so the Rule 11.9 honesty logic in
    run_afterclose (new>0 => ok) can be exercised.
    """
    calls = calls if calls is not None else []

    def runner(cmd, *, cwd=None, env_extra=None, timeout=None):
        calls.append(cmd)
        is_screener = any("screener.py" in str(c) for c in cmd)
        if is_screener:
            if screener_rc == 0 and snapshot_payload is not None:
                out_path = Path(cmd[cmd.index("--out") + 1])
                out_path.parent.mkdir(parents=True, exist_ok=True)
                out_path.write_text(json.dumps(snapshot_payload), encoding="utf-8")
            return screener_rc, "screener output", ""
        if any("emit_daily" in str(c) for c in cmd):
            return 0, (f"  new predictions:        {emit_new}\n"
                       f"  dropped (no close):     {emit_dropped}\n"
                       f"  skipped (already on disk): 0\n"), ""
        return 0, "swept", ""

    return runner, calls


def test_afterclose_happy_path_calls_screener_then_emit_then_sweep(tmp_path, monkeypatch):
    monkeypatch.setattr(dr, "SNAP_DIR", tmp_path / "screener")
    # no cohort for the month → the monthly guard runs emit + sweep at the tail
    monkeypatch.setattr(dr, "COHORT_DIR", tmp_path / "cohort_none")
    runner, calls = _fake_runner_factory(snapshot_payload={"asof": "2026-05-29",
                                                            "symbols": ["6966.T", "6768.T"]})
    rec = dr.run_afterclose(date(2026, 5, 30), runner=runner)

    assert rec["ok"] is True
    assert rec["candidates"]["ok"] is True
    assert rec["candidates"]["ticker_count"] == 2
    order = ["refresh" if any("refresh_htr_price_db" in str(c) for c in cmd) else
             "macro" if any("refresh_htr_macro_news" in str(c) for c in cmd) else
             "news" if any("refresh_htr_news" in str(c) for c in cmd) else
             "adr" if any("refresh_skhy_adr_watch" in str(c) for c in cmd) else
             "tdnet" if any("poll_tdnet_rss" in str(c) for c in cmd) else
             "revisions" if any("capture_tdnet_revisions" in str(c) for c in cmd) else
             "screener" if any("screener.py" in str(c) for c in cmd) else
             "meta" if any("refresh_ticker_metadata" in str(c) for c in cmd) else
             "skabu" if any("build_s_kabu_overlay" in str(c) for c in cmd) else
             "emit" if any("emit_daily" in str(c) for c in cmd) else
             "sweep" if any("sweep_pending" in str(c) for c in cmd) else
             "forward_eval" if any("forward_signal_report" in str(c) for c in cmd) else
             "cohort" if any("fundamental_cohort" in str(c) for c in cmd) else
             "value_livelog" if any("backtest_value_on_livelog" in str(c) for c in cmd) else
             "risk_mandate" if any("risk_mandate_snapshot" in str(c) for c in cmd) else "?"
             for cmd in calls]
    # price DB → news → macro → adr → TDnet corpus (ADR-0010/P17-4) → revision
    # docs (P23-A, perishable) → screener → meta → S株 overlay → emit → sweep →
    # forward shadow-eval (Rule 16) → monthly fundamental cohort (P19-02b) →
    # value-on-livelog early read (P23-F) → risk-mandate trace (P25-05,
    # Section 17) — all tail steps non-fatal, research-only/read-only.
    assert order == ["refresh", "news", "macro", "adr", "tdnet", "revisions",
                     "screener", "meta", "skabu", "emit", "sweep", "forward_eval",
                     "cohort", "cohort", "value_livelog", "risk_mandate"]


def test_afterclose_fail_closed_aborts_emit_when_screener_fails(tmp_path, monkeypatch):
    monkeypatch.setattr(dr, "SNAP_DIR", tmp_path / "screener")
    runner, calls = _fake_runner_factory(screener_rc=1)
    rec = dr.run_afterclose(date(2026, 5, 30), runner=runner)

    assert rec["ok"] is False
    assert rec["candidates"]["ok"] is False
    # emit / sweep MUST NOT run after a failed refresh.
    assert not any("emit_daily" in str(c) for cmd in calls for c in cmd)
    assert not any("sweep_pending" in str(c) for cmd in calls for c in cmd)


def test_afterclose_rejects_zero_candidate_snapshot(tmp_path, monkeypatch):
    monkeypatch.setattr(dr, "SNAP_DIR", tmp_path / "screener")
    runner, calls = _fake_runner_factory(snapshot_payload={"asof": "2026-05-29", "symbols": []})
    rec = dr.run_afterclose(date(2026, 5, 30), runner=runner)

    assert rec["ok"] is False
    assert "0 candidates" in rec["candidates"]["reason"]
    assert not any("emit_daily" in str(c) for cmd in calls for c in cmd)


def test_dry_run_executes_nothing(tmp_path, monkeypatch):
    monkeypatch.setattr(dr, "SNAP_DIR", tmp_path / "screener")
    called = []

    def runner(cmd, **kw):
        called.append(cmd)
        return 0, "", ""

    rec = dr.run_afterclose(date(2026, 5, 30), dry_run=True, runner=runner)
    assert rec["ok"] is True
    assert called == []
    assert "plan" in rec


def test_record_carries_no_execution_or_broker_fields(tmp_path, monkeypatch):
    monkeypatch.setattr(dr, "SNAP_DIR", tmp_path / "screener")
    runner, _ = _fake_runner_factory(snapshot_payload={"asof": "2026-05-29", "symbols": ["6966.T"]})
    rec = dr.run_afterclose(date(2026, 5, 30), runner=runner)
    blob = json.dumps(rec).lower()
    for forbidden in ("broker", "order_id", "account", "submit", "route", "execute"):
        assert forbidden not in blob


def test_preopen_reports_smoke_and_freshness(tmp_path, monkeypatch):
    monkeypatch.setattr(dr, "SNAP_DIR", tmp_path / "screener")

    def runner(cmd, **kw):
        return 0, "1266 passed, 5 deselected", ""

    rec = dr.run_preopen(date(2026, 5, 30), runner=runner)
    assert rec["mode"] == "preopen"
    assert rec["smoke"]["passed"] is True
    assert "passed" in rec["smoke"]["summary"]
    assert rec["candidate_snapshot_present"] is False  # none in tmp snap dir


def test_afterclose_zero_new_with_drops_is_honest_fail(tmp_path, monkeypatch):
    """Rule 11.9 — a green return code that collected 0 new samples (all dropped
    for missing close) MUST be ok=False and MUST NOT log 'emitted'. This is the
    2026-06-01 silent-no-op-with-success-log bug (Codex review)."""
    monkeypatch.setattr(dr, "SNAP_DIR", tmp_path / "screener")
    runner, _ = _fake_runner_factory(
        snapshot_payload={"asof": "2026-06-01", "symbols": ["6768.T", "6966.T"]},
        emit_new=0, emit_dropped=2)
    rec = dr.run_afterclose(date(2026, 6, 1), runner=runner)
    assert rec["ok"] is False
    assert rec["collection"]["new_predictions"] == 0
    assert rec["collection"]["dropped_no_close"] == 2
    assert not any("emitted" in n for n in rec["notes"])
    assert any("0 new forward samples" in n for n in rec["notes"])


def test_afterclose_idempotent_rerun_is_ok(tmp_path, monkeypatch):
    """All-skipped (already-on-disk) re-run is a legitimate ok (idempotent)."""
    monkeypatch.setattr(dr, "SNAP_DIR", tmp_path / "screener")
    runner, _ = _fake_runner_factory(
        snapshot_payload={"asof": "2026-05-29", "symbols": ["6768.T"]},
        emit_new=0, emit_dropped=0)
    # override emit to report all-skipped
    def runner2(cmd, *, cwd=None, env_extra=None, timeout=None):
        if any("screener.py" in str(c) for c in cmd):
            out_path = Path(cmd[cmd.index("--out") + 1]); out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps({"asof": "2026-05-29", "symbols": ["6768.T"]}), encoding="utf-8")
            return 0, "screener output", ""
        if any("emit_daily" in str(c) for c in cmd):
            return 0, "  new predictions:        0\n  dropped (no close):     0\n  skipped (already on disk): 1\n", ""
        return 0, "swept", ""
    rec = dr.run_afterclose(date(2026, 5, 30), runner=runner2)
    assert rec["ok"] is True
    assert any("idempotent" in n for n in rec["notes"])


def test_smoke_captures_error_tail_on_failure():
    """A failed pre-open smoke must record WHY (error tail), not just '275 errors'
    with no cause — the 2026-06-02 un-diagnosable failure."""
    def runner(cmd, **kw):
        return 1, "E   ImportError: boom\n1017 passed, 5 deselected, 275 errors in 33.99s\n", "traceback detail"
    res = dr.smoke(runner=runner)
    assert res["passed"] is False
    assert "275 errors" in res["summary"]
    assert "error_tail" in res and "ImportError" in res["error_tail"]


def test_smoke_pass_has_no_error_tail():
    def runner(cmd, **kw):
        return 0, "1292 passed, 5 deselected in 9s", ""
    res = dr.smoke(runner=runner)
    assert res["passed"] is True
    assert "error_tail" not in res


def test_smoke_command_uses_workspace_basetemp():
    """Windows scheduled runs must not depend on the user Temp pytest root, which
    can be locked or permission-denied. Keep temp files inside the HTR workspace.

    P37-03 step 5 moved the location: it used to be `.pytest_tmp/daily-smoke`,
    one of four scratch roots that had accumulated at the repo top level because
    every caller invented its own. It now comes from
    `common.runtime_paths.lane_paths`, so there is one owner and one layout.
    """
    from hot_theme_rotator.common.runtime_paths import lane_paths

    cmd = [str(part) for part in dr.SMOKE_CMD]
    joined = " ".join(cmd)
    assert "--basetemp" in cmd
    expected = lane_paths(dr.SMOKE_LANE, create=False)
    assert str(expected.basetemp) in joined
    assert str(expected.cache) in joined
    # Still inside the repo, still nowhere near the user Temp.
    assert ".runtime" in joined
    assert "AppData" not in joined


def test_smoke_subprocess_gets_a_pinned_temp_directory(monkeypatch):
    """Pinning --basetemp alone was never enough.

    pytest's own scratch moved into the workspace, but any library reaching for
    `tempfile` still landed in the system temp — the directory whose ACL defect
    produces the false hangs and mass collection ERRORs this lane exists to
    avoid. The environment is the half that was missing, so it is asserted here
    rather than left to a comment.
    """
    from hot_theme_rotator.common.runtime_paths import lane_paths

    captured: dict[str, object] = {}

    def fake_runner(cmd, *, cwd=None, env_extra=None, timeout=None):
        captured["env_extra"] = env_extra or {}
        return 0, "1 passed in 0.1s", ""

    dr.smoke(runner=fake_runner)
    env_extra = captured["env_extra"]
    expected_tmp = str(lane_paths(dr.SMOKE_LANE, create=False).tmp)
    for key in ("TMP", "TEMP", "TMPDIR"):
        assert env_extra.get(key) == expected_tmp, f"{key} not pinned into the workspace"
    assert env_extra.get("PYTHONNOUSERSITE") == "1"


def test_monthly_cohort_guard_emits_once_per_month(tmp_path, monkeypatch):
    # P19-02b: emit is guarded by an existing cohort for the month; sweep always runs.
    import tools.daily_routine as dr

    cohort_dir = tmp_path / "reports" / "research_cohorts" / "fundamental" / "predictions"
    cohort_dir.mkdir(parents=True)
    monkeypatch.setattr(dr, "COHORT_DIR", cohort_dir)

    calls = []
    def fake_runner(cmd, **kw):
        calls.append(cmd)
        return (0, "", "")

    from datetime import date
    # no cohort yet → emit + sweep
    out1 = dr._maybe_monthly_cohort(date(2026, 8, 3), runner=fake_runner)
    assert out1["emit_rc"] == 0 and out1["sweep_rc"] == 0
    verbs = [c[2] for c in calls if "fundamental_cohort.py" in str(c[1])]
    assert verbs == ["emit", "sweep"]

    # simulate the emitted cohort landing on disk
    (cohort_dir / "2026-08-03.jsonl").write_text("{}", encoding="utf-8")
    calls.clear()
    out2 = dr._maybe_monthly_cohort(date(2026, 8, 20), runner=fake_runner)
    assert out2["emit_rc"] == "skipped_month_exists"
    verbs2 = [c[2] for c in calls if "fundamental_cohort.py" in str(c[1])]
    assert verbs2 == ["sweep"]  # emit skipped, sweep still runs


def test_health_block_fallback_still_obeys_the_biconditional():
    """A crashed health reporter may not contradict `ok` either.

    Reporting `degraded` on an ok=False run would understate a real core
    failure at exactly the moment the diagnostic meant to explain it is the
    thing that broke.
    """
    import tools.daily_routine as dr

    def _boom(_record):
        raise RuntimeError("assessor exploded")

    original = dr.assess_record
    dr.assess_record = _boom
    try:
        failed = dr.health_block({"mode": "afterclose", "ok": False})
        degraded = dr.health_block({"mode": "afterclose", "ok": True})
    finally:
        dr.assess_record = original

    assert failed["health_status"] == "failed"
    assert degraded["health_status"] == "degraded"
    for block in (failed, degraded):
        # Rule 15.10.7 — never an aggregate without its components.
        assert [r["code"] for r in block["degraded_components"]] == [
            "health_reporter.exception"]
        assert "assessor exploded" in block["degraded_components"][0]["detail"]
