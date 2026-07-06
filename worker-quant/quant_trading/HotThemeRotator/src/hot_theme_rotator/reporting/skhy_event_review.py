"""SKHY forward-evidence review (P20-07 / Rule 11.15, Rule 16, ADR-0010).

Before the SKHY→Japan-semi overlay may EVER be promoted from shadow, it must show
repeatable, net-of-cost, LIVE-only forward evidence. This harness:

- counts distinct LIVE event-date clusters (same-day duplicates collapse to one,
  so a single noisy day cannot masquerade as many samples);
- excludes backdated clusters entirely (never pooled with live — the 2026-06-23
  Simpson's-paradox lesson);
- computes the cross-sectional Rank-IC of sympathy score vs forward relative
  return only when enough same-day breadth exists (reuses forward_signal_eval);
- carries the Rule 16.0 cost-hurdle context;
- defaults to ``insufficient_data`` and NEVER grants promotion itself — promotion
  is a separately-governed step under ADR-0010 / Rule 16.

No probability / win-rate / expected-return / edge field is produced.
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

DEFAULT_MIN_EVENT_CLUSTERS = 20
DEFAULT_COST_HURDLE = 0.04

__all__ = ["review_skhy_events", "DEFAULT_MIN_EVENT_CLUSTERS", "DEFAULT_COST_HURDLE"]


def review_skhy_events(
    event_clusters: Sequence[Mapping[str, Any]],
    *,
    min_event_clusters: int = DEFAULT_MIN_EVENT_CLUSTERS,
    cost_hurdle: float = DEFAULT_COST_HURDLE,
    min_names: int = 5,
) -> dict[str, Any]:
    """Review live SKHY event clusters; return a fail-closed verdict.

    Each cluster: ``{"date": ISO, "live": bool, "samples": [{"sympathy": float,
    "fwd_rel_return": float}, ...]}``. Returns a dict with ``verdict``,
    ``eventClusters``, ``minEventClusters``, ``rankIc5d``, ``costHurdle``,
    ``promotionAllowed`` (always False here), and ``notes``.
    """
    notes = ["live-only", "no probability", "requires ADR-0010 and Rule 16 gates"]
    # Fail-closed: a cluster counts as live evidence ONLY if it explicitly sets
    # live=True. Missing/None/False live is NOT live evidence (P20 fix#2).
    live = [c for c in event_clusters if c.get("live") is True]
    dates = {c.get("date") for c in live if c.get("date")}
    n_clusters = len(dates)
    base = {
        "eventClusters": n_clusters,
        "minEventClusters": min_event_clusters,
        "costHurdle": cost_hurdle,
        "promotionAllowed": False,  # promotion is a separate ADR-0010 / Rule 16 step
        "notes": notes,
    }
    if n_clusters < min_event_clusters:
        return {**base, "verdict": "insufficient_data", "rankIc5d": None}

    # Collapse same-day clusters into one panel per distinct date (no duplicate
    # same-day pseudo-samples), then require cross-sectional breadth per day.
    by_date: dict[str, list] = {}
    for c in live:
        d = c.get("date")
        if not d:
            continue
        by_date.setdefault(d, []).extend(c.get("samples", []) or [])

    daily: list[tuple[list[float], list[float]]] = []
    for d in sorted(by_date):
        rows = by_date[d]
        if len(rows) >= min_names:
            s = [float(x["sympathy"]) for x in rows]
            r = [float(x["fwd_rel_return"]) for x in rows]
            daily.append((s, r))

    rank_ic5d = None
    if daily:
        from hot_theme_rotator.backtesting.forward_signal_eval import rank_ic

        ric = rank_ic(daily, min_names=min_names)
        if ric.n_days > 0:
            rank_ic5d = round(ric.mean_ic, 4)

    verdict = "review" if rank_ic5d is not None else "insufficient_data"
    return {**base, "verdict": verdict, "rankIc5d": rank_ic5d}
