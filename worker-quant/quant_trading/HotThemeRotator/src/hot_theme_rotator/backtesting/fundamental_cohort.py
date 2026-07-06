"""Fundamental research-cohort lane (P19-02b — Rule 16.2 live-only forward).

The P23-B gate-passing family (earnings_yield / value_bp) was validated at a
63-day horizon on the ~3,000-name panel, but the §10 production forward log
only matures 1/3/5-day outcomes on ~50 momentum-screened names per day. This
lane accumulates the forward evidence the promotion decision actually needs —
WITHOUT touching the production pipeline:

- SEPARATE storage: ``reports/research_cohorts/fundamental/{predictions,outcomes}``.
  Nothing here enters the decision log, cohort review, or calibration (same
  separation discipline as the Rule 11.15 external ADR lane).
- Monthly broad-universe cohorts: every symbol with a PIT fundamental snapshot
  and a price gets a row with its earnings_yield / value_bp as of the cohort
  date (published strictly before — Rule 8.2).
- Maturity-honest sweep: 21D/63D forward returns fill in only once the window
  has actually closed; unmatured horizons stay ``None``, never estimated.
- Research-only: no probability, no advice, never promotes anything. The
  verdict consumer is the Rule 16 / ADR-0010 promotion review.
"""
from __future__ import annotations

import json
from bisect import bisect_right
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

HORIZONS = (21, 63)

LANE_DIR = ("reports", "research_cohorts", "fundamental")


def _lane(base_dir: Path | str, kind: str) -> Path:
    p = Path(base_dir).joinpath(*LANE_DIR, kind)
    p.mkdir(parents=True, exist_ok=True)
    return p


def build_cohort_rows(
    asof: str,
    *,
    symbols: Sequence[str],
    eps_lookup: Callable[[str, str], Optional[float]],
    bps_lookup: Callable[[str, str], Optional[float]],
    price_lookup: Callable[[str], Optional[float]],
) -> list[dict[str, Any]]:
    """Score the universe as of ``asof`` (PIT enforced by the injected lookups).

    A symbol needs a price and at least one fundamental to enter the cohort;
    a missing individual field stays ``None`` (honest, never imputed).
    """
    rows: list[dict[str, Any]] = []
    for sym in symbols:
        px = price_lookup(sym)
        if not px or px <= 0:
            continue
        eps = eps_lookup(sym, asof)
        bps = bps_lookup(sym, asof)
        if eps is None and bps is None:
            continue
        rows.append({
            "symbol": sym,
            "asof": asof,
            "ref_price": float(px),
            "earnings_yield": (float(eps) / float(px)) if eps is not None else None,
            "value_bp": (float(bps) / float(px)) if bps is not None and bps else None,
        })
    return rows


def emit_cohort(base_dir: Path | str, asof: str, rows: list[dict[str, Any]]) -> Path:
    """Write one cohort file; idempotent — an existing cohort is never rewritten
    (PIT: the scores as first recorded are the record)."""
    path = _lane(base_dir, "predictions") / f"{asof}.jsonl"
    if path.exists():
        return path
    with path.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    return path


def _forward_return(series, asof: str, horizon: int, today: str) -> Optional[float]:
    dates, closes = series
    hi = bisect_right(dates, today)  # only sessions ≤ sweep date are observable
    dates, closes = dates[:hi], closes[:hi]
    i = bisect_right(dates, asof)  # entry = first session strictly after asof
    if i >= len(dates) or i + horizon >= len(closes):
        return None  # window not closed yet — stays unmatured
    a, b = closes[i], closes[i + horizon]
    return (b / a - 1.0) if a and a > 0 else None


def sweep_cohorts(
    base_dir: Path | str,
    *,
    price_series: Callable[[str], Optional[tuple[list[str], list[float]]]],
    today: str,
) -> dict[str, Any]:
    """Fill 21D/63D realized returns for every cohort row whose window closed.

    Rewrites each cohort's outcome file wholesale (cheap; monthly files), so a
    re-sweep upgrades unmatured horizons without duplicating rows.
    ``price_series(symbol) -> (dates, closes) | None`` is injected (research
    price store in production; synthetic series in tests).
    """
    pred_dir = _lane(base_dir, "predictions")
    out_dir = _lane(base_dir, "outcomes")
    swept = 0
    for pf in sorted(pred_dir.glob("*.jsonl")):
        rows = [json.loads(ln) for ln in pf.read_text(encoding="utf-8").splitlines() if ln]
        out_rows = []
        for r in rows:
            series = price_series(r["symbol"])
            rec = {"symbol": r["symbol"], "asof": r["asof"], "swept_asof": today}
            for h in HORIZONS:
                rec[f"ret_{h}d"] = (
                    _forward_return(series, r["asof"], h, today) if series else None
                )
            out_rows.append(rec)
            swept += 1
        with (out_dir / pf.name).open("w", encoding="utf-8") as fh:
            for rec in out_rows:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return {"swept_rows": swept, "cohorts": len(list(pred_dir.glob('*.jsonl')))}


def cohort_report(base_dir: Path | str, *, min_names: int = 50) -> dict[str, Any]:
    """Per-signal, per-horizon cross-sectional Rank-IC over matured cohorts.

    Research diagnostic for the Rule 16 promotion review — descriptive only.
    """
    from hot_theme_rotator.backtesting.forward_signal_eval import rank_ic

    pred_dir = _lane(base_dir, "predictions")
    out_dir = _lane(base_dir, "outcomes")
    # cohort_date -> {symbol: (scores…)} joined with outcomes
    report: dict[str, Any] = {"n_cohorts": 0, "signals": {}}
    panels: dict[tuple[str, int], list[tuple[list[float], list[float]]]] = {}
    cohorts = 0
    for pf in sorted(pred_dir.glob("*.jsonl")):
        of = out_dir / pf.name
        if not of.exists():
            continue
        preds = {json.loads(ln)["symbol"]: json.loads(ln)
                 for ln in pf.read_text(encoding="utf-8").splitlines() if ln}
        outs = {json.loads(ln)["symbol"]: json.loads(ln)
                for ln in of.read_text(encoding="utf-8").splitlines() if ln}
        cohorts += 1
        for sig in ("earnings_yield", "value_bp"):
            for h in HORIZONS:
                pairs = []
                for sym, p in preds.items():
                    s = p.get(sig)
                    ret = (outs.get(sym) or {}).get(f"ret_{h}d")
                    if s is not None and ret is not None:
                        pairs.append((float(s), float(ret)))
                if len(pairs) >= min_names:
                    panels.setdefault((sig, h), []).append(
                        ([s for s, _ in pairs], [r for _, r in pairs]))
    report["n_cohorts"] = cohorts
    for (sig, h), daily in panels.items():
        res = rank_ic(daily)
        report["signals"].setdefault(sig, {})[str(h)] = {
            "mean_ic": res.mean_ic, "t_stat": res.t_stat, "n_cohorts": res.n_days,
        }
    return report
