"""Forward-test harness — live-only data loader + one-call signal summary.

Increment 2 of the forward-test harness (P19-01). Wires the pure stats core in
:mod:`forward_signal_eval` to the real decision-log journals and to the P17
``tradability`` cost model, so any signal can be put through the §16 gate with a
single call.

Rule 16.2 (live-only, no bootstrap pooling) is enforced here: only predictions
flagged live (``is_live_prediction``) are grouped; bootstrap/backdated samples
never enter signal-skill evaluation. Rule 16.0's cost ``c_rt`` is auto-derived
from each name's ``reference_price`` via the JPX tick ladder
(``round_trip_cost_bps``), not hand-filled, so the hurdle reflects the real
account's execution cost.

The grouping is split into a pure function (:func:`group_live_daily`, testable
with plain objects) and a thin IO wrapper (:func:`load_live_panels`).
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from statistics import median
from typing import Any, Callable, NamedTuple, Optional, Sequence

from hot_theme_rotator.backtesting.forward_signal_eval import (
    cost_hurdle,
    cross_sectional_dispersion,
    net_ic_after_cost,
    rank_ic,
)

__all__ = [
    "DayPanel",
    "group_live_daily",
    "estimate_round_trip_cost_frac",
    "load_live_panels",
    "summarize_live_signal",
]


class DayPanel(NamedTuple):
    """One trading day's cross-section: aligned scores / forward returns / prices."""

    date: str
    scores: list[float]
    returns: list[float]
    prices: list[Optional[float]]  # reference_price per name (for cost), may be None


def _ref_price(pred: Any) -> Optional[float]:
    extra = getattr(pred, "extra", {}) or {}
    raw = extra.get("reference_price")
    if raw is None:
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def group_live_daily(
    preds: Sequence[Any],
    outs_by_pid: dict,
    *,
    horizon: int,
    is_live: Callable[[Any], bool],
    min_names: int = 5,
) -> list[DayPanel]:
    """Pure: join live predictions to outcomes, group into per-day panels.

    A name enters a day's panel only if its prediction is live, an outcome
    exists, and that outcome has a realized return at ``{horizon}D``. Days with
    fewer than ``min_names`` usable names are dropped (a cross-sectional stat on
    a handful of names is noise). ``score`` is the prediction's ``buy`` field.
    """
    hk = f"{int(horizon)}D"
    byday: dict[str, list] = defaultdict(list)
    for p in preds:
        if is_live(p):
            byday[p.trade_date].append(p)

    panels: list[DayPanel] = []
    for date in sorted(byday):
        scores: list[float] = []
        returns: list[float] = []
        prices: list[Optional[float]] = []
        for p in byday[date]:
            out = outs_by_pid.get(p.prediction_id)
            if out is None:
                continue
            rr = getattr(out, "realized_returns", {}) or {}
            if hk not in rr or rr[hk] is None:
                continue
            scores.append(float(p.buy))
            returns.append(float(rr[hk]))
            prices.append(_ref_price(p))
        if len(scores) >= min_names:
            panels.append(DayPanel(date, scores, returns, prices))
    return panels


def estimate_round_trip_cost_frac(
    prices: Sequence[Optional[float]],
    *,
    spread_ticks: float = 3.0,
    slippage_bps: float = 0.0,
) -> Optional[float]:
    """Median tick-implied round-trip cost as a fraction (Rule 16.0 c_rt).

    Uses the P17 ``tradability.round_trip_cost_bps`` (JPX tick ladder). Names
    without a usable price are ignored. Returns None if no price is usable.
    """
    from hot_theme_rotator.candidate_engine.tradability import round_trip_cost_bps

    bps = [
        round_trip_cost_bps(px, spread_ticks=spread_ticks, slippage_bps=slippage_bps)
        for px in prices
        if px is not None and px > 0
    ]
    if not bps:
        return None
    return median(bps) / 1e4


def load_live_panels(
    base_dir: str | Path = ".",
    *,
    horizon: int,
    min_names: int = 5,
) -> list[DayPanel]:
    """IO wrapper: read live predictions+outcomes from journals, group by day.

    Only dates with BOTH a predictions and an outcomes file are read. Returns
    [] if the journals are absent (no error — empty is a legitimate state).
    """
    from hot_theme_rotator.calibration.isotonic_recalibrator import is_live_prediction
    from hot_theme_rotator.decision_log.jsonl_writer import (
        read_outcomes,
        read_predictions,
    )

    base = Path(base_dir)
    pdir = base / "reports" / "predictions"
    odir = base / "reports" / "outcomes"
    if not pdir.exists() or not odir.exists():
        return []
    dates = sorted(
        {p.stem for p in pdir.iterdir() if p.suffix == ".jsonl"}
        & {p.stem for p in odir.iterdir() if p.suffix == ".jsonl"}
    )
    preds: list = []
    outs: list = []
    for d in dates:
        preds += read_predictions(trade_date=d, base_dir=base)
        outs += read_outcomes(trade_date=d, base_dir=base)
    outs_by_pid = {o.prediction_id: o for o in outs}
    return group_live_daily(
        preds, outs_by_pid, horizon=horizon, is_live=is_live_prediction, min_names=min_names
    )


def summarize_live_signal(
    base_dir: str | Path = ".",
    *,
    horizons: Sequence[int] = (1, 3, 5),
    turnover: float = 0.7,
    round_trip_cost: Optional[float] = None,
    spread_ticks: float = 3.0,
    min_names: int = 5,
) -> dict:
    """One-call §16 gate: per-horizon Rank-IC, sigma_r, cost hurdle, net, clears?.

    ``round_trip_cost`` defaults to the median tick-implied cost over the live
    universe (Rule 16.0). ``clears`` is True only for a positively-signed signal
    that beats cost — a negative-IC reversal lead never clears as-is (ADR-0011).
    """
    out: dict = {"turnover": turnover, "horizons": {}}
    for H in horizons:
        panels = load_live_panels(base_dir, horizon=H, min_names=min_names)
        if not panels:
            out["horizons"][H] = {"n_days": 0}
            continue
        daily = [(p.scores, p.returns) for p in panels]
        dret = [p.returns for p in panels]
        ric = rank_ic(daily, min_names=min_names)
        sigma = cross_sectional_dispersion(dret, min_names=min_names)
        all_prices = [px for p in panels for px in p.prices]
        crt = (
            round_trip_cost
            if round_trip_cost is not None
            else estimate_round_trip_cost_frac(all_prices, spread_ticks=spread_ticks)
        )
        rec: dict = {
            "n_days": ric.n_days,
            "mean_ic": ric.mean_ic,
            "t_stat": ric.t_stat,
            "sigma_r": sigma,
            "round_trip_cost": crt,
        }
        if sigma > 0 and crt is not None:
            net = net_ic_after_cost(ric.mean_ic, sigma, turnover, crt)
            rec["hurdle"] = cost_hurdle(sigma, turnover, crt)
            rec["net_ic_after_cost"] = net
            rec["clears"] = ric.mean_ic > 0 and net > 0
        out["horizons"][H] = rec
    return out
