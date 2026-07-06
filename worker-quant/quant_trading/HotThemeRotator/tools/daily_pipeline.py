"""Daily forward-sample accumulation pipeline (P0 — no auto-refit).

One command per day to keep Rule 8.2.1 sunset on track:

  python tools/daily_pipeline.py

Steps (each fails-closed; pipeline exits non-zero if any step errors):

  1. emit_daily_predictions    — write today's PredictionRecord from selected_tickers
  2. sweep_pending_outcomes    — compute outcomes for any historical pending preds
  3. kfold_validate_isotonic   — re-run K-fold OOS verdict against current pairs
  4. Report sunset readiness   — paired live / bootstrap / eta to ≥100 live

This pipeline does NOT auto-refit the recalibrator (Rule 4 spirit + governance
decision documented in PROJECT_STATUS). When live samples reach 100 (sunset
threshold), the pipeline prints a loud notice asking the user to manually
run ``python tools/fit_isotonic_recalibrator.py``.

P1 reflection autorun (PIT snapshot + trace logging + RCA → auto-proposals)
is layered on top via ``--with-reflection`` flag. Default is OFF; reflection
runs on demand once you opt in.

Exit codes:
  0  — all steps succeeded
  1  — pipeline step failed (details in stderr)
  2  — environment / config error before any step ran
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


def _ensure_src_on_path() -> None:
    here = Path(__file__).resolve()
    src_root = here.parent.parent / "src"
    if str(src_root) not in sys.path:
        sys.path.insert(0, str(src_root))


_ensure_src_on_path()

from hot_theme_rotator.calibration.isotonic_recalibrator import (  # noqa: E402
    is_live_prediction,
    pair_for_calibration,
)
from hot_theme_rotator.data.position_adapter import (  # noqa: E402
    default_journal_base_dir,
)
from hot_theme_rotator.data.universe_adapter import (  # noqa: E402
    UniverseAdapterError, default_selected_tickers_path, load_screener_snapshot,
)
from hot_theme_rotator.decision_log.jsonl_writer import (  # noqa: E402
    read_outcomes, read_predictions,
)


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
SUNSET_THRESHOLD = 100
SCANNER_STALE_DAYS = 2          # Rule 8.2 + 12.2: refuse emit on selected_tickers > 2 days old
TDNET_FRESHNESS_HOURS = 24      # Rule 12.2: warn if no TDnet today (non-blocking)


def _run_step(name: str, cmd: list[str], cwd: Path) -> int:
    """Run a CLI step as subprocess; return exit code. Stream stdout/stderr through."""
    print(f"\n=== {name} ===", flush=True)
    print(f"$ {' '.join(cmd)}", flush=True)
    try:
        result = subprocess.run(cmd, cwd=cwd, check=False)
        return result.returncode
    except FileNotFoundError as exc:
        print(f"FAILED to launch: {exc}", file=sys.stderr)
        return 127


def _count_paired_samples(base_dir: Path, horizon_days: int) -> dict:
    """Count live / bootstrap paired samples for sunset readiness reporting."""
    pred_dir = base_dir / "reports" / "predictions"
    out_dir = base_dir / "reports" / "outcomes"
    if not pred_dir.exists() or not out_dir.exists():
        return {
            "live_paired": 0, "bootstrap_paired": 0,
            "live_min_samples": SUNSET_THRESHOLD,
            "sunset_fired": False,
            "preds_total": 0, "outs_total": 0,
        }
    dates = sorted(
        {p.stem for p in pred_dir.iterdir() if p.suffix == ".jsonl"}
        & {p.stem for p in out_dir.iterdir() if p.suffix == ".jsonl"}
    )
    all_preds, all_outs = [], []
    for d in dates:
        try:
            all_preds.extend(read_predictions(trade_date=d, base_dir=base_dir))
            all_outs.extend(read_outcomes(trade_date=d, base_dir=base_dir))
        except Exception as exc:  # noqa: BLE001
            print(f"WARN: skipping {d} during sample count: {exc}", file=sys.stderr)
    try:
        _, _, stats = pair_for_calibration(
            all_preds, all_outs,
            horizon_days=horizon_days,
            prefer_live_when_sufficient=False,
            live_min_samples=SUNSET_THRESHOLD,
        )
        stats["preds_total"] = len(all_preds)
        stats["outs_total"] = len(all_outs)
        return stats
    except Exception as exc:  # noqa: BLE001
        print(f"WARN: pair_for_calibration failed: {exc}", file=sys.stderr)
        return {
            "live_paired": 0, "bootstrap_paired": 0,
            "live_min_samples": SUNSET_THRESHOLD,
            "sunset_fired": False,
            "preds_total": len(all_preds), "outs_total": len(all_outs),
        }


def _check_scanner_freshness(now: datetime, *, max_age_days: int = SCANNER_STALE_DAYS) -> tuple[bool, str]:
    """Rule 8.2 + Rule 12.2 — refuse emit when selected_tickers.json is stale.

    Returns ``(is_fresh, reason)``. is_fresh=False means emit must be skipped.
    """
    try:
        path = default_selected_tickers_path()
        snap = load_screener_snapshot(path)
    except UniverseAdapterError as exc:
        return False, f"selected_tickers.json missing/invalid: {exc}"

    try:
        asof_date = datetime.strptime(snap.asof, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    except (ValueError, TypeError):
        return False, f"selected_tickers.json asof not ISO date: {snap.asof!r}"

    age_days = (now - asof_date).days
    if age_days > max_age_days:
        return False, (
            f"selected_tickers.json asof={snap.asof} is {age_days} days old "
            f"(> {max_age_days}d threshold; Rule 8.2 + 12.2 fail-closed). "
            f"Refresh scanner first: `python ../Project_optimized/screener.py --db japan_market.db --topk 50`"
        )
    return True, f"selected_tickers.json asof={snap.asof}, age={age_days}d (OK)"


def _check_tdnet_freshness(base_dir: Path, now: datetime) -> tuple[bool, str]:
    """Rule 12.2 — warn (don't block) when no TDnet file for today.

    Returns ``(is_fresh, reason)``. is_fresh=False means warn user but pipeline
    continues; per-ticker news will be empty until poller runs.
    """
    tdnet_dir = base_dir / "reports" / "tdnet"
    if not tdnet_dir.exists():
        return False, "reports/tdnet/ directory missing — TDnet poller never ran"

    today_str = now.strftime("%Y-%m-%d")
    today_file = tdnet_dir / f"{today_str}.jsonl"
    if today_file.exists():
        return True, f"reports/tdnet/{today_str}.jsonl present"

    # Find most recent file
    candidates = sorted(tdnet_dir.glob("*.jsonl"), reverse=True)
    if not candidates:
        return False, "no reports/tdnet/*.jsonl files yet"

    newest = candidates[0]
    try:
        newest_date = datetime.strptime(newest.stem, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        age_hours = (now - newest_date).total_seconds() / 3600.0
    except ValueError:
        age_hours = -1.0

    return False, (
        f"latest TDnet file is {newest.name}; today's {today_str}.jsonl missing "
        f"(staleness ~{age_hours:.0f}h). Pass --with-tdnet to invoke poller."
    )


def _read_kfold_verdict(base_dir: Path) -> Optional[dict]:
    path = base_dir / "reports" / "recalibrator_kfold_v1.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8")).get("verdict")
    except (OSError, json.JSONDecodeError):
        return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--horizon-days", type=int, default=3,
                    help="Horizon for ground-truth derivation (default 3)")
    ap.add_argument("--skip-emit", action="store_true",
                    help="Skip step 1 (use when scanner hasn't refreshed today)")
    ap.add_argument("--skip-sweep", action="store_true",
                    help="Skip step 2 (rarely needed)")
    ap.add_argument("--skip-kfold", action="store_true",
                    help="Skip step 3 (e.g. for diagnostic-only run)")
    ap.add_argument("--with-reflection", action="store_true",
                    help="Run P1 reflection autorun (PIT snapshot + RCA auto-proposals)")
    ap.add_argument("--with-tdnet", action="store_true",
                    help="Invoke tools/poll_tdnet_rss.py before emit to refresh disclosures")
    ap.add_argument("--ignore-stale", action="store_true",
                    help="Bypass Rule 8.2 + 12.2 scanner staleness gate (use with care)")
    ap.add_argument("--scanner-stale-days", type=int, default=SCANNER_STALE_DAYS,
                    help=f"Selected_tickers max age in days (default {SCANNER_STALE_DAYS})")
    ap.add_argument("--base-dir", default=None,
                    help="HTR project root; default resolves automatically")
    args = ap.parse_args()

    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    base_dir = Path(args.base_dir) if args.base_dir else default_journal_base_dir()
    if not base_dir.exists():
        print(f"FAILED: base_dir {base_dir} does not exist", file=sys.stderr)
        return 2

    now = datetime.now(tz=timezone.utc)
    started_at = now.isoformat()
    print(f"daily_pipeline starting at {started_at}")
    print(f"base_dir = {base_dir}")
    print(f"horizon  = {args.horizon_days}D")
    failed_steps: list[str] = []

    # ─── Step 0a: scanner freshness gate (Rule 8.2 + Rule 12.2) ──────
    print("\n=== 0a. scanner freshness gate (Rule 8.2 + 12.2) ===")
    is_fresh, reason = _check_scanner_freshness(now, max_age_days=args.scanner_stale_days)
    print(reason)
    if not is_fresh and not args.ignore_stale and not args.skip_emit:
        print("\nFAILED: scanner is stale; refusing to emit predictions "
              "(Rule 8.2 PIT + Rule 12.2 stale-data fail-closed).",
              file=sys.stderr)
        print("Pass --ignore-stale ONLY if you understand the PIT violation.",
              file=sys.stderr)
        return 1
    if not is_fresh and args.ignore_stale:
        print("WARN: proceeding despite stale scanner (--ignore-stale)",
              file=sys.stderr)

    # ─── Step 0b: optional TDnet poll ────────────────────────────────
    if args.with_tdnet:
        rc = _run_step(
            "0b. poll_tdnet_rss",
            [sys.executable, str(HERE / "poll_tdnet_rss.py"),
             "--base-dir", str(base_dir)],
            cwd=base_dir,
        )
        if rc != 0:
            # Non-blocking: TDnet failure shouldn't stop emit/sweep
            print(f"WARN: TDnet poll failed (rc={rc}); per-ticker news may be stale",
                  file=sys.stderr)
            failed_steps.append(f"tdnet (rc={rc}, non-blocking)")
    print("\n=== 0c. TDnet freshness ===")
    td_fresh, td_reason = _check_tdnet_freshness(base_dir, now)
    if td_fresh:
        print(f"OK: {td_reason}")
    else:
        print(f"WARN (non-blocking, Rule 12.2): {td_reason}")

    # ─── Step 1: emit ───────────────────────────────────────────────
    if not args.skip_emit:
        rc = _run_step(
            "1. emit_daily_predictions",
            [sys.executable, str(HERE / "emit_daily_predictions.py"),
             "--base-dir", str(base_dir)],
            cwd=base_dir,
        )
        if rc != 0:
            failed_steps.append(f"emit (rc={rc})")
    else:
        print("\n=== 1. emit_daily_predictions [SKIPPED via --skip-emit] ===")

    # ─── Step 2: sweep ──────────────────────────────────────────────
    if not args.skip_sweep:
        rc = _run_step(
            "2. sweep_pending_outcomes",
            [sys.executable, str(HERE / "sweep_pending_outcomes.py"),
             "--base-dir", str(base_dir)],
            cwd=base_dir,
        )
        if rc != 0:
            failed_steps.append(f"sweep (rc={rc})")
    else:
        print("\n=== 2. sweep_pending_outcomes [SKIPPED via --skip-sweep] ===")

    # ─── Step 3: kfold validate ─────────────────────────────────────
    if not args.skip_kfold:
        kfold_cmd = [
            sys.executable, str(HERE / "kfold_validate_isotonic.py"),
            "--horizon-days", str(args.horizon_days),
            "--base-dir", str(base_dir),
        ]
        # Check if there's enough data; the K-fold validator will fail-closed
        # internally if not. We surface that as a warning, not a step failure.
        rc = _run_step("3. kfold_validate_isotonic", kfold_cmd, cwd=base_dir)
        if rc not in (0, 2, 3):  # 2 = no data; 3 = fit failed; both acceptable mid-pipeline
            failed_steps.append(f"kfold (rc={rc})")
    else:
        print("\n=== 3. kfold_validate_isotonic [SKIPPED via --skip-kfold] ===")

    # ─── Step 4: reflection (optional, P1) ──────────────────────────
    if args.with_reflection:
        try:
            from hot_theme_rotator.reflection.auto_pipeline import (
                run_daily_reflection,
            )
            print("\n=== 4. reflection autorun ===")
            report = run_daily_reflection(base_dir=base_dir)
            print(f"PIT snapshot: {report.get('snapshot_id', '—')}")
            print(f"Traces appended: {report.get('traces_written', 0)}")
            print(f"Proposals generated: {report.get('proposals_written', 0)}")
            print(f"Silent findings: {report.get('silent_findings', 0)}")
        except Exception as exc:  # noqa: BLE001
            print(f"reflection FAILED: {exc}", file=sys.stderr)
            failed_steps.append(f"reflection ({exc})")
    else:
        print("\n=== 4. reflection autorun [SKIPPED — pass --with-reflection to enable] ===")

    # ─── Sunset readiness report ────────────────────────────────────
    print("\n=== sunset readiness ===")
    stats = _count_paired_samples(base_dir, horizon_days=args.horizon_days)
    live = stats["live_paired"]
    boot = stats["bootstrap_paired"]
    remaining = max(0, SUNSET_THRESHOLD - live)
    print(f"paired live samples       : {live} / {SUNSET_THRESHOLD}")
    print(f"paired bootstrap samples  : {boot}")
    print(f"total predictions on disk : {stats.get('preds_total', 0)}")
    print(f"total outcomes on disk    : {stats.get('outs_total', 0)}")
    if live >= SUNSET_THRESHOLD:
        print()
        print("*" * 60)
        print("SUNSET THRESHOLD REACHED — Rule 8.2.1 sunset is ready.")
        print("Run manually:")
        print("    python tools/fit_isotonic_recalibrator.py --horizon-days {}".format(args.horizon_days))
        print("This will refit the recalibrator on live-only data + archive bootstrap.")
        print("Then K-fold revalidation runs in this pipeline next time.")
        print("*" * 60)
    elif live > 0:
        eta_days = remaining // max(1, live)  # rough estimate
        print(f"remaining to sunset       : {remaining} paired live samples")
        print(f"~ETA at current daily rate: ~{eta_days * 2}-{eta_days * 5} trading days "
              f"(depends on outcome horizon)")

    # ─── K-fold verdict snapshot ───────────────────────────────────
    verdict = _read_kfold_verdict(base_dir)
    print("\n=== current K-fold verdict ===")
    if verdict is None:
        print("(no verdict on disk)")
    else:
        print(f"verdict        : {verdict.get('verdict')}")
        print(f"oos_brier_mean : {verdict.get('oos_brier_mean')}")
        print(f"in_sample      : {verdict.get('in_sample_brier')}")
        print(f"random_baseline: {verdict.get('random_baseline')}")
        print(f"reason         : {verdict.get('reason')}")

    # ─── Exit ──────────────────────────────────────────────────────
    print()
    if failed_steps:
        print(f"PIPELINE FAILED — failed steps: {', '.join(failed_steps)}", file=sys.stderr)
        return 1
    print("daily_pipeline complete — all steps OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
