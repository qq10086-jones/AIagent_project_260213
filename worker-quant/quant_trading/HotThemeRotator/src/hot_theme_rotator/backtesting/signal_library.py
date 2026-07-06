"""Signal library + one-call gate runner (ADR-0011 / §16, P19-02).

A *signal* is a pure function from per-name decision records to one score per
name. :func:`evaluate_signal` runs ANY signal function through the forward-test
gate (:mod:`forward_signal_eval`) on live-only data, returning per-horizon
Rank-IC, cost hurdle, and clears? — so a new idea is measured before it is
trusted (Rule 16.0 break-even, 16.1 Rank-IC, 16.2 live-only). Default eval cost
is the S株 zero-fee + 5bps timing buffer (Rule 5.2), the cheapest real mode.

Seed signals:
- ``reversal_of_score``: −buy. Tests the Phase-1 mean-reversion lead. NOT an
  independent signal (same information reversed) — present to validate the
  plumbing and to track the lead forward. Does NOT clear the promotion bar.
- (planned) ``price_reversal_5d`` / ``low_vol``: need the kline price adapter.
- (planned, blocked) ``disclosure_drift``: needs the TDnet corpus (ADR-0010).
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Callable, Mapping, NamedTuple, Optional, Sequence

from hot_theme_rotator.backtesting.forward_signal_eval import (
    cost_hurdle,
    cross_sectional_dispersion,
    net_ic_after_cost,
    rank_ic,
)

# Default evaluation cost: SBI S株 is commission+spread free (Rule 5.2); only a
# ~5bps session-reference timing buffer remains.
S_KABU_COST = 0.0005

__all__ = [
    "NameDayRecord",
    "SignalFn",
    "reversal_of_score",
    "make_price_reversal_signal",
    "kline_prior_return_lookup",
    "cross_signal_rank_corr",
    "group_scored_daily",
    "evaluate_signal",
    "SIGNALS",
]


class NameDayRecord(NamedTuple):
    """One (name, decision-day) record — the input a signal scores."""

    prediction_id: str
    symbol: str
    trade_date: str
    buy: float
    reference_price: Optional[float]


SignalFn = Callable[[Sequence[NameDayRecord]], Mapping[str, float]]


def reversal_of_score(records: Sequence[NameDayRecord]) -> dict[str, float]:
    """Seed #1: negate the screener buy-score (the Phase-1 reversal lead).

    NOT independent of the existing score — it is the same information read
    backwards. Tracked as a hypothesis, not a tradable signal.
    """
    return {r.prediction_id: -r.buy for r in records}


def group_scored_daily(
    records: Sequence[NameDayRecord],
    scores: Mapping[str, float],
    returns_by_pid: Mapping[str, float],
    *,
    min_names: int = 5,
) -> tuple[list[tuple[list[float], list[float]]], list[list[float]]]:
    """Pure: build per-day (scores, returns) panels for names that have BOTH a
    signal score and a forward return. Days below ``min_names`` are dropped.
    Returns ``(daily_pairs, daily_returns)`` ready for the harness core.
    """
    byday: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for r in records:
        s = scores.get(r.prediction_id)
        ret = returns_by_pid.get(r.prediction_id)
        if s is None or ret is None:
            continue
        byday[r.trade_date].append((float(s), float(ret)))
    daily: list[tuple[list[float], list[float]]] = []
    dret: list[list[float]] = []
    for d in sorted(byday):
        rows = byday[d]
        if len(rows) >= min_names:
            daily.append(([s for s, _ in rows], [ret for _, ret in rows]))
            dret.append([ret for _, ret in rows])
    return daily, dret


def _load_records_and_returns(base_dir, horizon: int):
    from hot_theme_rotator.calibration.isotonic_recalibrator import is_live_prediction
    from hot_theme_rotator.decision_log.jsonl_writer import (
        read_outcomes,
        read_predictions,
    )

    base = Path(base_dir)
    pdir = base / "reports" / "predictions"
    odir = base / "reports" / "outcomes"
    if not pdir.exists() or not odir.exists():
        return [], {}
    dates = sorted(
        {p.stem for p in pdir.iterdir() if p.suffix == ".jsonl"}
        & {p.stem for p in odir.iterdir() if p.suffix == ".jsonl"}
    )
    preds: list = []
    outs: list = []
    for d in dates:
        preds += read_predictions(trade_date=d, base_dir=base)
        outs += read_outcomes(trade_date=d, base_dir=base)
    ob = {o.prediction_id: o for o in outs}
    hk = f"{int(horizon)}D"
    records: list[NameDayRecord] = []
    ret_by_pid: dict[str, float] = {}
    for p in preds:
        if not is_live_prediction(p):
            continue
        extra = getattr(p, "extra", {}) or {}
        rp = extra.get("reference_price")
        try:
            rp = float(rp) if rp is not None else None
        except (TypeError, ValueError):
            rp = None
        records.append(NameDayRecord(p.prediction_id, p.symbol, p.trade_date, float(p.buy), rp))
        o = ob.get(p.prediction_id)
        rr = (getattr(o, "realized_returns", {}) or {}) if o else {}
        if o and hk in rr and rr[hk] is not None:
            ret_by_pid[p.prediction_id] = float(rr[hk])
    return records, ret_by_pid


def evaluate_signal(
    signal_fn: SignalFn,
    base_dir: str | Path = ".",
    *,
    horizons: Sequence[int] = (1, 3, 5),
    turnover: float = 0.7,
    round_trip_cost: float = S_KABU_COST,
    min_names: int = 5,
) -> dict:
    """Run any signal function through the §16 gate on live-only data.

    Returns per-horizon Rank-IC, t-stat, sigma_r, cost hurdle, net, clears?.
    ``clears`` requires a positively-signed signal whose net beats cost — and is
    NECESSARY, not SUFFICIENT, for promotion (Rule 16.6 still defers to the
    ADR-0010 anti-overfit gate: Harvey t≥3, Deflated Sharpe, PBO, forward log).
    """
    name = getattr(signal_fn, "__name__", str(signal_fn))
    out: dict = {"signal": name, "round_trip_cost": round_trip_cost, "horizons": {}}
    for H in horizons:
        records, ret_by_pid = _load_records_and_returns(base_dir, H)
        if not records:
            out["horizons"][H] = {"n_days": 0}
            continue
        scores = signal_fn(records)
        daily, dret = group_scored_daily(records, scores, ret_by_pid, min_names=min_names)
        if not daily:
            out["horizons"][H] = {"n_days": 0}
            continue
        ric = rank_ic(daily)
        sig = cross_sectional_dispersion(dret)
        rec: dict = {
            "mean_ic": ric.mean_ic,
            "t_stat": ric.t_stat,
            "n_days": ric.n_days,
            "sigma_r": sig,
            "round_trip_cost": round_trip_cost,
        }
        if sig > 0:
            net = net_ic_after_cost(ric.mean_ic, sig, turnover, round_trip_cost)
            rec["hurdle"] = cost_hurdle(sig, turnover, round_trip_cost)
            rec["net_ic_after_cost"] = net
            rec["clears"] = ric.mean_ic > 0 and net > 0
        out["horizons"][H] = rec
    return out


def make_price_reversal_signal(*, lookback: int = 5, return_lookup) -> SignalFn:
    """Seed #2 (independent): score = −(prior ``lookback``-session return).

    High recent return → low score (bet on short-term reversal). This is
    derived from PRICE history, not the screener score, so it can be a genuinely
    orthogonal stack member (check via :func:`cross_signal_rank_corr`, Rule 16.3).
    ``return_lookup(symbol, trade_date) -> float | None`` supplies the PIT prior
    return (injected so this stays unit-testable without a DB).
    """
    def signal(records: Sequence[NameDayRecord]) -> dict[str, float]:
        out: dict[str, float] = {}
        for r in records:
            pr = return_lookup(r.symbol, r.trade_date)
            if pr is not None:
                out[r.prediction_id] = -float(pr)
        return out

    signal.__name__ = f"price_reversal_{int(lookback)}d"
    return signal


def kline_prior_return_lookup(db_path=None, *, lookback: int = 5, calendar_buffer: int = 12):
    """Build a point-in-time prior-return lookup from the daily-prices DB.

    Uses closes STRICTLY before ``trade_date`` (no look-ahead — Rule 8.2). Returns
    None when there are fewer than ``lookback+1`` usable closes. IO is isolated
    here so the signal itself (above) stays pure/testable.
    """
    from datetime import date as _date, timedelta as _td

    from hot_theme_rotator.data.kline_adapter import (
        LegacyDailyPriceFetcher,
        default_db_path,
    )

    path = db_path or default_db_path()
    fetcher = LegacyDailyPriceFetcher(path)

    def lookup(symbol: str, trade_date: str) -> Optional[float]:
        td = _date.fromisoformat(trade_date)
        end = (td - _td(days=1)).isoformat()  # strictly before decision date
        start = (td - _td(days=lookback * 2 + calendar_buffer)).isoformat()
        try:
            bars = list(fetcher.fetch(symbol=symbol, start_date=start, end_date=end))
        except Exception:
            return None
        closes = [float(b.close) for b in bars if getattr(b, "close", None) not in (None, 0)]
        if len(closes) < lookback + 1:
            return None
        prev, base = closes[-1], closes[-1 - lookback]
        if base <= 0:
            return None
        return prev / base - 1.0

    return lookup


def make_fundamental_yield_signal(*, field: str, fundamental_lookup) -> SignalFn:
    """P23-B gate-passing family: score = PIT per-share fundamental ÷ reference price.

    ``field`` ∈ {'eps' → earnings_yield, 'bps' → value_bp}. Historical verdict
    (2026-07-03): earnings_yield@63D DSR 0.992 (first gate pass in project
    history), value_bp close behind, split-half confirmed. This wiring is the
    Rule 16.2 live-only forward track those signals must now walk — it never
    promotes anything by itself. ``fundamental_lookup(symbol, trade_date) ->
    float | None`` supplies the PIT value (injected, unit-testable without a DB);
    the record's own emit-time ``reference_price`` is the denominator.
    """
    name = {"eps": "earnings_yield", "bps": "value_bp"}[field]

    def signal(records: Sequence[NameDayRecord]) -> dict[str, float]:
        out: dict[str, float] = {}
        for r in records:
            if not r.reference_price or float(r.reference_price) <= 0:
                continue
            v = fundamental_lookup(r.symbol, r.trade_date)
            if v is not None:
                out[r.prediction_id] = float(v) / float(r.reference_price)
        return out

    signal.__name__ = name
    return signal


_FUND_FIELDS = {"eps": "eps", "bps": "bps"}


def fundamentals_pit_lookup(field: str, db_path=None):
    """Point-in-time per-share fundamental lookup from htr_fundamentals.db.

    Uses ONLY ``period_basis='reported'`` rows (values stamped with their
    ORIGINAL filing datetime) published STRICTLY before ``trade_date``
    (Rule 8.2 — no look-ahead). Loads once into memory (~13k reported rows);
    returns None when no snapshot predates the decision date.
    """
    import sqlite3
    from bisect import bisect_left

    col = _FUND_FIELDS[field]  # whitelist — never interpolate caller input
    path = db_path or (
        Path(__file__).resolve().parents[3] / "data" / "raw" / "htr_fundamentals.db"
    )
    per: dict[str, list[tuple[str, float]]] = defaultdict(list)
    if Path(path).exists():
        conn = sqlite3.connect(str(path))
        try:
            for sym, pub, val in conn.execute(
                f"select symbol, published_ts, {col} from fundamental_snapshots "
                f"where period_basis='reported' and {col} is not null"
            ):
                per[sym].append((str(pub)[:10], float(val)))
        finally:
            conn.close()
    table = {}
    for sym, rows in per.items():
        rows.sort()
        table[sym] = ([p for p, _ in rows], [v for _, v in rows])

    def lookup(symbol: str, trade_date: str) -> Optional[float]:
        entry = table.get(symbol)
        if not entry:
            return None
        pubs, vals = entry
        i = bisect_left(pubs, str(trade_date)[:10]) - 1  # strictly before
        return vals[i] if i >= 0 else None

    return lookup


def cross_signal_rank_corr(
    records: Sequence[NameDayRecord],
    scores_a: Mapping[str, float],
    scores_b: Mapping[str, float],
    *,
    min_names: int = 5,
) -> Optional[float]:
    """Mean per-day Spearman between two score maps — the Rule 16.3 orthogonality
    check. |ρ| < 0.5 is required before two signals may stack."""
    from hot_theme_rotator.backtesting.forward_signal_eval import spearman

    byday: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for r in records:
        a = scores_a.get(r.prediction_id)
        b = scores_b.get(r.prediction_id)
        if a is None or b is None:
            continue
        byday[r.trade_date].append((float(a), float(b)))
    rhos = []
    for rows in byday.values():
        if len(rows) >= min_names:
            rho = spearman([x for x, _ in rows], [y for _, y in rows])
            if rho is not None:
                rhos.append(rho)
    return sum(rhos) / len(rhos) if rhos else None


# Registry — new signals plug in here once they clear Rule 16.0 in research.
# (price_reversal is constructed on demand because it needs a price lookup.)
SIGNALS: dict[str, SignalFn] = {
    "reversal_of_score": reversal_of_score,
}
