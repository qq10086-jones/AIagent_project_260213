"""HTR Local Beta v0 daily routine orchestrator (Rule 15.5).

Deterministic, LLM-free, no-GPU, advice-only. Chains the *automatable* parts of
the Monday operating rhythm so the operator only has to watch the dashboard.

Two modes:

- ``--mode preopen``   : run the daily smoke gate (``pytest -m "not slow"``) and a
                         candidate-freshness check; log readiness.
- ``--mode afterclose``: refresh candidates (the deterministic sibling screener →
                         HTR-native snapshot, ``--no_db_write`` per ADR-0005) →
                         emit forward predictions → sweep outcomes; log.

Governance:

- Never touches broker / order / paper / live execution, and never calls an LLM /
  GPU path (Rule 3, Rule 15.4, user "no background LLM" preference). It shells out
  only to the four deterministic tools hard-listed below.
- Fail-closed: a failed or empty candidate refresh ABORTS emit — no stale or
  fabricated predictions (Rule 8.2 / 12.2).
- Idempotent: re-running the same trading day is a no-op (emit prediction_id hash,
  sweep outcome_id). Rule 15.4: emits only for a genuine just-closed session, never
  an accelerated historical replay.

Two things this routine deliberately does NOT automate (they cannot be):
  * recording a fill — only happens when the operator actually trades externally;
  * LLM narrative briefs — pulled on demand under the no-background-LLM rule.

Output: one structured JSON line per run appended to
``reports/observability/daily_routine_log.jsonl``.

Usage::

  python tools/daily_routine.py --mode afterclose            # latest trading day
  python tools/daily_routine.py --mode afterclose --asof 2026-05-29
  python tools/daily_routine.py --mode preopen
  python tools/daily_routine.py --mode afterclose --dry-run  # plan only, no writes
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Callable

# cp932 / non-UTF-8 Windows consoles cannot encode em-dashes or Japanese in
# progress output. Make stdout/stderr UTF-8 safe so a scheduled run never crashes.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    except (AttributeError, ValueError):
        pass

HERE = Path(__file__).resolve().parent          # tools/
ROOT = HERE.parent                              # HotThemeRotator/
SIBLING = ROOT.parent / "Project_optimized"
SCREENER = SIBLING / "screener.py"
SIBLING_DB = SIBLING / "japan_market.db"
# P12-03(a)+ — the screener now reads an HTR-OWNED price DB (a one-time sibling
# snapshot + yfinance-appended fresh days), so candidates reflect recent moves
# even while the sibling EOD feed is frozen. The sibling stays read-only (ADR-0005).
HTR_DB = ROOT / "data" / "raw" / "htr_market.db"
DB_DEFAULT = HTR_DB
REFRESH_DB = HERE / "refresh_htr_price_db.py"
REFRESH_NEWS = HERE / "refresh_htr_news.py"
REFRESH_MACRO = HERE / "refresh_htr_macro_news.py"
REFRESH_META = HERE / "refresh_ticker_metadata.py"
BUILD_SKABU = HERE / "build_s_kabu_overlay.py"   # task#7 / Rule 5.2 S株 overlay producer
REFRESH_ADR = HERE / "refresh_skhy_adr_watch.py" # P20-03 / Rule 11.15 external ADR watch
POLL_TDNET = HERE / "poll_tdnet_rss.py"  # ADR-0010 / P17-4: accumulate disclosure corpus
CAPTURE_REVISIONS = HERE / "capture_tdnet_revisions.py"  # P23-A: download PERISHABLE revision PDFs (TDnet 31-day window) + parse magnitudes
SNAP_DIR = ROOT / "reports" / "screener"
LOG_PATH = ROOT / "reports" / "observability" / "daily_routine_log.jsonl"
EMIT = HERE / "emit_daily_predictions.py"
SWEEP = HERE / "sweep_pending_outcomes.py"
FORWARD_EVAL = HERE / "forward_signal_report.py"  # §16/ADR-0010 shadow-eval (non-fatal, research-only)
FUNDAMENTAL_COHORT = HERE / "fundamental_cohort.py"  # P19-02b: monthly value/quality forward cohort
COHORT_DIR = ROOT / "reports" / "research_cohorts" / "fundamental" / "predictions"
VALUE_LIVELOG = HERE / "backtest_value_on_livelog.py"  # P23-F: early value read on the live log (63D matures ~Aug 26)

# JPX trading calendar (Rule 15.4 fail-closed) lives in the src layer so it is
# reusable + unit-tested; add src to path the same way emit/sweep do.
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
from hot_theme_rotator.data.jpx_calendar import (  # noqa: E402
    calendar_covers,
    is_trading_day,  # noqa: F401  (re-exported for callers/tests)
    latest_trading_day,
)

SMOKE_CMD = [sys.executable, "-m", "pytest", "tests", "-m", "not slow", "-q",
             "-p", "no:cacheprovider", "--basetemp", str(ROOT / ".pytest_tmp" / "daily-smoke")]

# Runner signature: (cmd, cwd, env_extra, timeout) -> (returncode, stdout, stderr).
Runner = Callable[..., "tuple[int, str, str]"]


def _run(cmd: list[str], *, cwd: str | None = None,
         env_extra: dict[str, str] | None = None,
         timeout: int | None = None) -> tuple[int, str, str]:
    """Real subprocess runner — UTF-8 forced so sibling CJK output never crashes."""
    env = os.environ.copy()
    env.setdefault("PYTHONIOENCODING", "utf-8")
    if env_extra:
        env.update(env_extra)
    proc = subprocess.run(
        cmd, cwd=cwd, env=env, text=True, capture_output=True,
        encoding="utf-8", errors="replace", timeout=timeout,
    )
    return proc.returncode, proc.stdout, proc.stderr


def _now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def refresh_candidates(asof: date, db_path: Path, *, runner: Runner = _run) -> dict[str, Any]:
    """Run the deterministic sibling screener → HTR-native snapshot (read-only).

    ``--no_db_write`` keeps it read-only against the sibling (ADR-0005). Returns a
    fail-closed dict: ``ok`` is False on non-zero exit, unreadable output, or zero
    tickers — the caller MUST NOT emit when ``ok`` is False.
    """
    SNAP_DIR.mkdir(parents=True, exist_ok=True)
    out = SNAP_DIR / f"selected_tickers_{asof.isoformat()}.json"
    # Refresh the HTR-native price DB first (sibling snapshot + yfinance fresh days)
    # so the screener sees recent sessions. Non-fatal: on failure the screener still
    # runs on the last-known HTR DB; the screener's own zero-candidate guard remains
    # the fail-closed gate (P12-03a+).
    rc_refresh, _ro, err_refresh = runner(
        [sys.executable, str(REFRESH_DB), "--target-date", asof.isoformat(),
         "--db", str(db_path), "--sibling", str(SIBLING_DB)],
        cwd=str(ROOT), env_extra={"PYTHONIOENCODING": "utf-8"}, timeout=400)
    # Refresh the HTR-native news timeline (Google News JP) so the dashboard +
    # theme engine see fresh catalysts, not the frozen sibling news tables.
    # Non-fatal: a news fetch failure never blocks candidate screening.
    rc_news, _nn, _ne = runner(
        [sys.executable, str(REFRESH_NEWS), "--asof", asof.isoformat(), "--base-dir", str(ROOT)],
        cwd=str(ROOT), env_extra={"PYTHONIOENCODING": "utf-8"}, timeout=300)
    # Refresh the HTR-native MACRO overlay (fx/monetary/geopolitics, P16-E2b) so the
    # dashboard's macro layer no longer freezes (was stale since 2026-05-30). The
    # Event Desk + macro card consume it. Non-fatal: a macro fetch failure never
    # blocks candidate screening.
    rc_macro, _mm, _me2 = runner(
        [sys.executable, str(REFRESH_MACRO), "--asof", asof.isoformat(), "--base-dir", str(ROOT)],
        cwd=str(ROOT), env_extra={"PYTHONIOENCODING": "utf-8"}, timeout=300)
    # P20-03 (Rule 11.15): refresh the EXTERNAL ADR watch (SKHY / 000660.KS / MU / NVDA /
    # SOXX / USDJPY). Read-only catalyst lane — non-fatal, NEVER blocks JP candidate
    # generation; a stale/pending SKHY leaves Japan ranking unchanged.
    rc_adr, _ad, _ae = runner(
        [sys.executable, str(REFRESH_ADR), "--asof", asof.isoformat(),
         "--out-dir", str(ROOT / "reports" / "adr")],
        cwd=str(ROOT), env_extra={"PYTHONIOENCODING": "utf-8"}, timeout=120)
    # ADR-0010 / P17-4: accumulate the TDnet 適時開示 corpus — disclosure-drift in
    # small/low-coverage names is the project's primary literature-backed edge thesis
    # (Jinushi 2023), but the corpus was never accumulating (poll unwired). Non-fatal:
    # a poll failure never blocks screening; PIT discipline is enforced at eval time.
    rc_tdnet, _tp, _te2 = runner(
        [sys.executable, str(POLL_TDNET), "--date", asof.isoformat()],
        cwd=str(ROOT), env_extra={"PYTHONIOENCODING": "utf-8"}, timeout=180)
    # P23-A: the RSS poll above stores only metadata (title/url). The revision
    # DOCUMENTS are perishable — TDnet's public site serves them for only ~31
    # days, so the magnitudes must be captured daily or they are lost forever.
    # Non-fatal, research-only; runs after the poll (consumes its corpus rows).
    rc_rev, _rv, _re2 = runner(
        [sys.executable, str(CAPTURE_REVISIONS), "--days", "35"],
        cwd=str(ROOT), env_extra={"PYTHONIOENCODING": "utf-8"}, timeout=600)
    cmd = [sys.executable, str(SCREENER), "--db", str(db_path),
           "--asof", asof.isoformat(), "--out", str(out), "--no_db_write"]
    try:
        rc, _stdout, stderr = runner(
            cmd, cwd=str(SIBLING), env_extra={"PYTHONIOENCODING": "utf-8"}, timeout=300)
    except subprocess.TimeoutExpired:
        return {"ok": False, "reason": "screener timeout (300s)", "out": str(out),
                "htr_db_refresh_rc": rc_refresh}
    if rc != 0:
        return {"ok": False, "reason": f"screener exit {rc}", "stderr": stderr[-500:],
                "out": str(out)}
    try:
        data = json.loads(out.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return {"ok": False, "reason": f"snapshot unreadable: {exc}", "out": str(out)}
    symbols = data.get("symbols") or data.get("tickers") or []
    if not symbols:
        return {"ok": False, "reason": "screener produced 0 candidates", "out": str(out)}
    # Refresh ticker metadata (yfinance sector/industry) for today's candidates so the
    # news-catalyst engine (ADR-0009) keeps mapping new names to themes. Non-fatal,
    # incremental (only unseen symbols fetched). Runs after the screener (needs symbols).
    rc_meta, _mn, _me = runner(
        [sys.executable, str(REFRESH_META), "--asof", asof.isoformat(), "--base-dir", str(ROOT)],
        cwd=str(ROOT), env_extra={"PYTHONIOENCODING": "utf-8"}, timeout=300)
    # task#7 (Rule 5.2): emit the S株 overlay snapshot — held/watchlist names the
    # 100-share-lot gate excludes but S株 unlocks. Non-fatal: never blocks screening.
    rc_skabu, _sk, _ske = runner(
        [sys.executable, str(BUILD_SKABU), "--asof", asof.isoformat(), "--base-dir", str(ROOT)],
        cwd=str(ROOT), env_extra={"PYTHONIOENCODING": "utf-8"}, timeout=120)
    return {"ok": True, "out": str(out), "ticker_count": len(symbols),
            "trade_date": data.get("asof") or asof.isoformat(),
            "htr_db_refresh_rc": rc_refresh, "news_refresh_rc": rc_news,
            "macro_refresh_rc": rc_macro, "meta_refresh_rc": rc_meta,
            "s_kabu_overlay_rc": rc_skabu, "adr_refresh_rc": rc_adr,
            "tdnet_poll_rc": rc_tdnet, "revision_capture_rc": rc_rev}


def _parse_emit_counts(text: str) -> dict[str, "int | None"]:
    """Pull emit's machine-checkable counts out of its stdout (Rule 11.9 honesty —
    a green return code is NOT proof a sample was collected)."""
    def grab(pat: str) -> "int | None":
        m = re.search(pat, text or "")
        return int(m.group(1)) if m else None
    return {
        "new_predictions": grab(r"new predictions:\s*(\d+)"),
        "dropped_no_close": grab(r"dropped \(no close\):\s*(\d+)"),
        "skipped_on_disk": grab(r"skipped \(already on disk\):\s*(\d+)"),
    }


def collect(snapshot_path: str, *, runner: Runner = _run) -> dict[str, Any]:
    """emit forward predictions from the snapshot, then sweep matured outcomes."""
    rc_e, out_e, err_e = runner(
        [sys.executable, str(EMIT), "--snapshot", snapshot_path],
        cwd=str(ROOT), timeout=300)
    rc_s, out_s, err_s = runner(
        [sys.executable, str(SWEEP)], cwd=str(ROOT), timeout=300)
    return {
        "emit_rc": rc_e, "emit_tail": (out_e or err_e)[-400:],
        "sweep_rc": rc_s, "sweep_tail": (out_s or err_s)[-400:],
        **_parse_emit_counts(out_e or ""),
    }


def smoke(*, runner: Runner = _run) -> dict[str, Any]:
    """Run the daily smoke lane (fast, no slow/numba). Pass == returncode 0."""
    rc, out, err = runner(SMOKE_CMD, cwd=str(ROOT), timeout=600)
    summary = ""
    for line in reversed((out or "").splitlines()):
        if any(tok in line for tok in ("passed", "failed", "error")):
            summary = line.strip()
            break
    result: dict[str, Any] = {"passed": rc == 0, "rc": rc, "summary": summary}
    if rc != 0:
        # Capture a diagnostic tail so a failed pre-open smoke is debuggable later
        # (the 2026-06-02 08:30 run logged "275 errors" with no cause). Errors
        # vs failures + the last lines distinguish a real regression from a
        # transient env/lock hiccup (which re-running clears).
        joined = ((out or "") + "\n" + (err or "")).splitlines()
        result["error_tail"] = "\n".join(joined[-30:])[-1500:]
    return result


def append_log(record: dict[str, Any]) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def run_preopen(today: date, *, runner: Runner = _run) -> dict[str, Any]:
    sm = smoke(runner=runner)
    asof = latest_trading_day(today)
    snap = SNAP_DIR / f"selected_tickers_{asof.isoformat()}.json"
    record = {
        "ts": _now_iso(), "mode": "preopen", "asof": asof.isoformat(),
        "ok": sm["passed"], "smoke": sm,
        "candidate_snapshot_present": snap.exists(),
        "notes": ["advice-only research cockpit; no execution; no LLM"],
    }
    return record


def _cohort_exists_this_month(asof: date) -> bool:
    ym = asof.strftime("%Y-%m")
    return COHORT_DIR.exists() and any(
        p.stem.startswith(ym) for p in COHORT_DIR.glob("*.jsonl"))


def _maybe_monthly_cohort(asof: date, *, runner: Runner = _run) -> dict[str, Any]:
    """Emit a fundamental cohort at most once per calendar month, then sweep.

    Emit is skipped (not re-run) once a cohort for the month exists, so the
    monthly cadence is guarded by the on-disk record — no separate scheduler.
    Sweep runs every afterclose to mature 21D/63D returns as windows close."""
    out: dict[str, Any] = {}
    if not _cohort_exists_this_month(asof):
        rc_e, _oe, _ee = runner(
            [sys.executable, str(FUNDAMENTAL_COHORT), "emit", "--asof", asof.isoformat()],
            cwd=str(ROOT), env_extra={"PYTHONIOENCODING": "utf-8"}, timeout=900)
        out["emit_rc"] = rc_e
    else:
        out["emit_rc"] = "skipped_month_exists"
    rc_s, _os, _es = runner(
        [sys.executable, str(FUNDAMENTAL_COHORT), "sweep", "--asof", asof.isoformat()],
        cwd=str(ROOT), env_extra={"PYTHONIOENCODING": "utf-8"}, timeout=1200)
    out["sweep_rc"] = rc_s
    return out


def run_afterclose(today: date, *, asof: date | None = None, db_path: Path | None = None,
                   dry_run: bool = False, runner: Runner = _run) -> dict[str, Any]:
    asof = asof or latest_trading_day(today)
    db_path = db_path or DB_DEFAULT
    record: dict[str, Any] = {
        "ts": _now_iso(), "mode": "afterclose", "asof": asof.isoformat(),
        "dry_run": dry_run,
        # Rule 15.4 honesty — when asof's year is outside the JPX holiday table,
        # holiday-skipping degrades to weekend-only and the data guard is the only
        # backstop; surface that so a future-year run isn't silently over-trusted.
        "calendar_covered": calendar_covers(asof),
        "notes": [],
    }
    if dry_run:
        record["ok"] = True
        record["plan"] = [
            f"screener --asof {asof.isoformat()} --no_db_write -> HTR snapshot",
            "emit_daily_predictions --snapshot <snapshot>",
            "sweep_pending_outcomes",
        ]
        record["notes"].append("dry-run: no subprocess executed, nothing written")
        return record

    cand = refresh_candidates(asof, db_path, runner=runner)
    record["candidates"] = cand
    if not cand.get("ok"):
        record["ok"] = False
        record["notes"].append("fail-closed: candidate refresh failed; emit ABORTED")
        return record

    coll = collect(cand["out"], runner=runner)
    record["collection"] = coll
    # Non-fatal forward shadow-eval (Rule 16.2 / ADR-0010): accumulate the live
    # IC + Deflated-Sharpe trace for candidate signals (price_reversal). Runs after
    # the sweep so the latest matured outcomes are included. Research-only — never
    # touches score_status/advice and never fails the collection (Rule 16.6).
    try:
        rc_f, out_f, err_f = runner(
            [sys.executable, str(FORWARD_EVAL), "--asof", asof.isoformat()],
            cwd=str(ROOT), timeout=300)
        record["forward_eval"] = {"rc": rc_f, "tail": (out_f or err_f or "")[-400:]}
    except Exception as exc:  # a diagnostic must never block collection
        record["forward_eval"] = {"rc": None, "error": str(exc)[:200]}
    # P19-02b: monthly fundamental cohort (the 63D forward track for the
    # gate-passing value/quality family). Emit at most once per calendar month
    # (network-heavy), then sweep to mature 21D/63D returns. Non-fatal,
    # research-only — this is the accumulation engine the forward verdict needs.
    try:
        record["cohort"] = _maybe_monthly_cohort(asof, runner=runner)
    except Exception as exc:  # never block collection
        record["cohort"] = {"rc": None, "error": str(exc)[:200]}
    # P23-F: early value read on the live log — accumulates daily and starts
    # emitting the 63D earnings_yield IC automatically once windows mature
    # (~Aug 26). Non-fatal, research-only; never promotes (Rule 16.6).
    try:
        rc_vll, out_vll, err_vll = runner(
            [sys.executable, str(VALUE_LIVELOG), "--asof", asof.isoformat(), "--write"],
            cwd=str(ROOT), env_extra={"PYTHONIOENCODING": "utf-8"}, timeout=300)
        record["value_livelog"] = {"rc": rc_vll, "tail": (out_vll or err_vll or "")[-300:]}
    except Exception as exc:
        record["value_livelog"] = {"rc": None, "error": str(exc)[:200]}
    new, dropped, skipped = coll.get("new_predictions"), coll.get("dropped_no_close"), coll.get("skipped_on_disk")
    procs_ok = (coll["emit_rc"] == 0 and coll["sweep_rc"] == 0)
    # Rule 11.9 honesty: a green return code with ZERO new samples is NOT a
    # successful collection. Only ok when emit wrote new rows, or the day was
    # already collected (idempotent re-run). Never log "emitted" on a 0-collect.
    if not procs_ok:
        record["ok"] = False
        record["notes"].append(f"fail: emit_rc={coll['emit_rc']} sweep_rc={coll['sweep_rc']}")
    elif new and new > 0:
        record["ok"] = True
        record["notes"].append(
            f"{new} new forward predictions emitted (dropped_no_close={dropped}, skipped={skipped}); outcomes swept")
    elif (skipped and skipped > 0) and not (dropped and dropped > 0):
        record["ok"] = True
        record["notes"].append(f"already collected for {asof.isoformat()} (idempotent, skipped={skipped}); no new samples")
    else:
        record["ok"] = False
        record["notes"].append(
            f"fail-closed: 0 new forward samples for {asof.isoformat()} "
            f"(dropped_no_close={dropped}, skipped={skipped}) — candidate closes not in daily_prices "
            f"(universe gap / EOD not finalized); nothing collected this run.")
    return record


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["preopen", "afterclose"], required=True)
    ap.add_argument("--asof", help="Override trade date (ISO). Default: latest weekday.")
    ap.add_argument("--db", help=f"Price DB path. Default: {DB_DEFAULT}")
    ap.add_argument("--dry-run", action="store_true",
                    help="afterclose: plan only, no subprocess / no writes")
    args = ap.parse_args(argv)

    today = date.today()
    asof = date.fromisoformat(args.asof) if args.asof else None
    db_path = Path(args.db) if args.db else None

    if args.mode == "preopen":
        record = run_preopen(today)
    else:
        record = run_afterclose(today, asof=asof, db_path=db_path, dry_run=args.dry_run)

    append_log(record)
    print(json.dumps(record, ensure_ascii=False, indent=2))
    return 0 if record.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
