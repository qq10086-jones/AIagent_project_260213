"""Policy Replay Engine (P11-03, ADR-0007 Layer 3).

Off-policy evaluation: given a recorded scanner output set + alternative
config grid, compute what each cell would have decided. Output Pareto frontier
in (pnl_proxy, miss_rate, alert_spam) space.

NOT Pearl do-calculus (Codex critique): we do not have an explicit causal
graph. We do **policy mutation** over recorded outputs — replace the live
threshold with a candidate threshold and re-derive the alert decision.

PIT validation: any ``recorded_output.available_ts > snapshot.decision_cutoff``
is a future-look — reject the entire replay.

Data Freshness Gate (Codex 2026-05-26 amendment): if the OHLC data backing
the recorded outcomes has a max-asof more than ``freshness_threshold_days``
calendar days before ``today``, the replay's validity class is set to
``data_too_stale`` regardless of other completeness. No numeric claim may be
published in that case.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Mapping, Sequence


__all__ = [
    "PolicyConfig",
    "PolicyReplayError",
    "PolicyReplayResult",
    "RecordedScannerOutput",
    "ReplayCellResult",
    "compute_pareto_frontier",
    "data_freshness_gate",
    "replay_under_policy_grid",
]


_JST = timezone(timedelta(hours=9), name="JST")


class PolicyReplayError(ValueError):
    """Raised on PIT violation or malformed replay input."""


@dataclass(frozen=True)
class RecordedScannerOutput:
    """One symbol's recorded state at the snapshot cutoff."""

    symbol: str
    raw_score: float           # opportunity score before any threshold cut
    intraday_move_pct: float   # for chase filter check
    is_in_cooling_off: bool    # already inside cooling-off window
    available_ts: str          # ISO ts the feature became available (PIT check)


@dataclass(frozen=True)
class PolicyConfig:
    """Mutable config dimensions for replay."""

    chase_threshold_pct: float = 10.0
    alert_budget_per_day: int = 10
    cooling_off_hours: float = 24.0
    scanner_threshold: float = 50.0


@dataclass(frozen=True)
class ReplayCellResult:
    """One cell of the replay grid: outcome under one ``PolicyConfig``."""

    config: PolicyConfig
    n_alerts: int
    n_alerts_dropped_chase: int
    n_alerts_dropped_cooling_off: int
    n_alerts_dropped_budget: int
    pnl_proxy: float           # Σ realized returns for alerted symbols
    miss_rate: float           # missed-good / total-good in actual outcomes
    alert_spam: float          # alerted-bad / total-alerts (1.0 if zero alerts)
    alerted_symbols: tuple[str, ...]


@dataclass(frozen=True)
class PolicyReplayResult:
    snapshot_id: str
    decision_cutoff: str
    data_max_asof: str
    freshness_threshold_days: int
    counterfactual_validity: str
    cells: tuple[ReplayCellResult, ...]
    pareto_frontier: tuple[ReplayCellResult, ...]


def data_freshness_gate(
    *,
    data_max_asof: str,
    now_date: str,
    threshold_days: int,
) -> tuple[bool, int]:
    """Return (ok, stale_days). ``ok=False`` means data is too stale.

    Patch M11 (Codex review #3, 2026-05-26): reject ``data_max_asof > now_date``
    explicitly — future-dated OHLC metadata is a PIT violation pretending to
    be very-fresh data, and would silently pass the freshness check.
    """
    if threshold_days <= 0:
        raise PolicyReplayError(
            f"freshness_threshold_days must be positive, got {threshold_days}"
        )
    try:
        d_data = date.fromisoformat(data_max_asof)
        d_now = date.fromisoformat(now_date)
    except (TypeError, ValueError) as exc:
        raise PolicyReplayError(
            f"data_max_asof and now_date must be ISO YYYY-MM-DD"
        ) from exc
    stale_days = (d_now - d_data).days
    if stale_days < 0:
        raise PolicyReplayError(
            f"data_max_asof {data_max_asof} is AFTER now_date {now_date} "
            f"(stale_days={stale_days}); future-dated data is a PIT violation"
        )
    return stale_days <= threshold_days, stale_days


def replay_under_policy_grid(
    *,
    snapshot_id: str,
    decision_cutoff: str,
    recorded_outputs: Sequence[RecordedScannerOutput],
    actual_outcomes: Mapping[str, float],
    config_grid: Sequence[PolicyConfig],
    data_max_asof: str,
    base_validity_class: str = "exact_replay",
    freshness_threshold_days: int = 5,
    now_date: str | None = None,
) -> PolicyReplayResult:
    """Run a policy grid replay over recorded outputs.

    ``base_validity_class`` is what the caller asserts the replay quality
    would be IF data freshness passes. ``data_too_stale`` overrides this when
    the freshness gate fails. PIT violation raises ``PolicyReplayError``
    BEFORE freshness check so the caller sees the bigger problem first.
    """
    if not recorded_outputs:
        raise PolicyReplayError("recorded_outputs must be non-empty")
    if not config_grid:
        raise PolicyReplayError("config_grid must be non-empty")
    # Patch M12 (Codex review #3): validate base_validity_class against enum.
    from hot_theme_rotator.observability.schema import VALIDITY_CLASSES
    if base_validity_class not in VALIDITY_CLASSES:
        raise PolicyReplayError(
            f"base_validity_class must be one of {VALIDITY_CLASSES}, "
            f"got {base_validity_class!r}"
        )

    # PIT discipline: every input feature must have been available at or
    # before the decision cutoff. A single violation aborts the entire replay.
    cutoff_dt = _parse_ts_tz(decision_cutoff, "decision_cutoff")
    for ro in recorded_outputs:
        feat_dt = _parse_ts_tz(ro.available_ts, f"{ro.symbol}.available_ts")
        if feat_dt > cutoff_dt:
            raise PolicyReplayError(
                f"PIT violation: feature for {ro.symbol} available_ts "
                f"{ro.available_ts} > decision_cutoff {decision_cutoff}"
            )

    # Freshness gate
    now_date = now_date or datetime.now(tz=_JST).date().isoformat()
    fresh_ok, stale_days = data_freshness_gate(
        data_max_asof=data_max_asof,
        now_date=now_date,
        threshold_days=freshness_threshold_days,
    )
    if fresh_ok:
        validity = base_validity_class
    else:
        validity = "data_too_stale"

    cells: list[ReplayCellResult] = []
    for cfg in config_grid:
        cells.append(_evaluate_cell(cfg, recorded_outputs, actual_outcomes))

    pareto = compute_pareto_frontier(cells) if validity != "data_too_stale" else ()

    return PolicyReplayResult(
        snapshot_id=snapshot_id,
        decision_cutoff=decision_cutoff,
        data_max_asof=data_max_asof,
        freshness_threshold_days=freshness_threshold_days,
        counterfactual_validity=validity,
        cells=tuple(cells),
        pareto_frontier=pareto,
    )


def compute_pareto_frontier(
    cells: Sequence[ReplayCellResult],
) -> tuple[ReplayCellResult, ...]:
    """Return the subset of cells not dominated by any other.

    Dominance: A dominates B iff A.pnl_proxy >= B.pnl_proxy AND
    A.miss_rate <= B.miss_rate AND A.alert_spam <= B.alert_spam, with at
    least one strict inequality. Cells are returned in input order.
    """
    frontier: list[ReplayCellResult] = []
    for i, c in enumerate(cells):
        dominated = False
        for j, other in enumerate(cells):
            if i == j:
                continue
            if _dominates(other, c):
                dominated = True
                break
        if not dominated:
            frontier.append(c)
    return tuple(frontier)


# ─── internals ──────────────────────────────────────────────────────────────


def _dominates(a: ReplayCellResult, b: ReplayCellResult) -> bool:
    if a.pnl_proxy < b.pnl_proxy:
        return False
    if a.miss_rate > b.miss_rate:
        return False
    if a.alert_spam > b.alert_spam:
        return False
    return (
        a.pnl_proxy > b.pnl_proxy
        or a.miss_rate < b.miss_rate
        or a.alert_spam < b.alert_spam
    )


def _evaluate_cell(
    cfg: PolicyConfig,
    outputs: Sequence[RecordedScannerOutput],
    actual_outcomes: Mapping[str, float],
) -> ReplayCellResult:
    """Apply policy thresholds to recorded outputs; compute outcome metrics."""
    alerted: list[str] = []
    dropped_chase = 0
    dropped_cool = 0
    dropped_budget = 0

    for ro in outputs:
        if ro.raw_score < cfg.scanner_threshold:
            continue
        if ro.intraday_move_pct >= cfg.chase_threshold_pct:
            dropped_chase += 1
            continue
        # cooling_off_hours = 0 disables; positive value uses recorded flag.
        if cfg.cooling_off_hours > 0 and ro.is_in_cooling_off:
            dropped_cool += 1
            continue
        if len(alerted) >= cfg.alert_budget_per_day:
            dropped_budget += 1
            continue
        alerted.append(ro.symbol)

    pnl_proxy = sum(float(actual_outcomes.get(s, 0.0)) for s in alerted)

    # miss_rate over the actual_outcomes universe: a symbol with positive
    # outcome that we did NOT alert is a miss. miss_rate = misses / total_good.
    total_good = sum(1 for v in actual_outcomes.values() if float(v) > 0)
    misses = sum(
        1 for s, v in actual_outcomes.items()
        if float(v) > 0 and s not in alerted
    )
    miss_rate = misses / total_good if total_good > 0 else 0.0

    # alert_spam: fraction of alerts that landed on non-positive outcomes.
    if not alerted:
        alert_spam = 1.0  # zero alerts is maximum spam by convention (also "no signal")
    else:
        bad = sum(1 for s in alerted if float(actual_outcomes.get(s, 0.0)) <= 0)
        alert_spam = bad / len(alerted)

    return ReplayCellResult(
        config=cfg,
        n_alerts=len(alerted),
        n_alerts_dropped_chase=dropped_chase,
        n_alerts_dropped_cooling_off=dropped_cool,
        n_alerts_dropped_budget=dropped_budget,
        pnl_proxy=pnl_proxy,
        miss_rate=miss_rate,
        alert_spam=alert_spam,
        alerted_symbols=tuple(alerted),
    )


def _parse_ts_tz(value: str, name: str) -> datetime:
    normalized = value.replace("Z", "+00:00") if isinstance(value, str) else value
    try:
        parsed = datetime.fromisoformat(normalized)
    except (TypeError, ValueError) as exc:
        raise PolicyReplayError(f"{name} must be ISO 8601 with timezone, got {value!r}") from exc
    if parsed.tzinfo is None:
        raise PolicyReplayError(f"{name} must carry timezone, got naive {value!r}")
    return parsed
