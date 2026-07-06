"""Disclosure-drift review harness (ADR-0010 P17-4) — the end-to-end pipeline.

Ties the P17 pieces into one honest, PIT-faithful evaluation of the inferred edge:

    TDnet disclosure events
      → surprise/novelty signal (P17-2: material + directional + novel)
      → PIT gate (drop events with no Japanese release timestamp — no look-ahead)
      → tradability gate (P17-1: affordable + cost-clearing for the account)
      → forward drift in the signal's direction, NET of round-trip cost
      → anti-overfit promotion gate (P17-3: Deflated Sharpe vs trials-aware noise)

It NEVER claims edge on its own. With an empty/sparse corpus (the current state — the
TDnet poller exists but the production disclosure corpus has not accrued) it returns
`verdict = "insufficient_data"`, accounting for every excluded event (never silently
dropped). A promotion verdict is computed ONLY when matured events ≥ `min_events` AND
the caller supplies the honest trial parameters; even then the append-only forward log
remains the final arbiter (Rule 8.2.2 / ADR-0010). Injected `event_return_fn` keeps it
deterministic and PIT-safe under test.
"""
from __future__ import annotations

from typing import Any, Callable

from hot_theme_rotator.candidate_engine.disclosure_surprise import rank_disclosures
from hot_theme_rotator.candidate_engine.tradability import tradability

# event_return_fn(event_dict, horizon_days) -> raw forward return (decimal) or None.
# PIT CONTRACT (Codex): the caller MUST compute this ONLY from prices available strictly
# AFTER the event's Japanese release/first-seen time, using the post-disclosure tradable
# entry price (not a pre-event or same-bar close), and MUST return None for any horizon
# not yet matured. The harness cannot see inside this fn — PIT cleanliness is the caller's
# guarantee; `event["price"]` is likewise expected to be the post-disclosure entry price.
EventReturnFn = Callable[[dict[str, Any], int], "float | None"]

DISCLOSURE = (
    "披露漂移评估:事件→惊奇信号→PIT 闸(无日文发布时间戳即剔除,防 look-ahead)→可交易闸"
    "(账户可成交+净成本)→按方向的净漂移→反过拟合准入闸。不自证 edge;语料不足时如实"
    "返回 insufficient_data,排除项逐类计数不静默丢;最终裁决仍归 append-only forward log。"
)


def build_disclosure_drift_review(
    events: list[dict[str, Any]],
    *,
    event_return_fn: EventReturnFn,
    account_jpy: float = 400_000.0,
    horizons: tuple[int, ...] = (3, 5, 10),
    min_events: int = 20,
    max_round_trip_bps: float = 60.0,
) -> dict[str, Any]:
    """Run the gated pipeline over `events` (each: title, published_ts, price[, category, adv_jpy]).

    Returns descriptive net-of-cost drift per horizon + an exclusion ledger + a verdict.
    `verdict` is "insufficient_data" until EVERY horizon has matured events ≥ `min_events`
    (gated by the slowest horizon). The anti-overfit promotion gate (Deflated Sharpe / PBO)
    is intentionally NOT auto-run here: it requires the true cross-trial Sharpe dispersion
    from a real trial matrix (`overfit_gate.promote_gate` over the searched configs), not a
    single per-event review — running it with a scalar here would be misleading (Codex)."""
    ranked = rank_disclosures(events)
    excluded = {"not_pit": 0, "not_material": 0, "no_direction": 0, "not_tradable": 0}
    actionable: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for ev in ranked:
        if not ev.get("pitOk"):
            excluded["not_pit"] += 1; continue
        if not ev.get("material"):
            excluded["not_material"] += 1; continue
        if ev.get("direction", 0) == 0:
            excluded["no_direction"] += 1; continue
        price = ev.get("price")
        adv = ev.get("adv_jpy")
        # ADV gate is enforced when the event carries volume; otherwise structural-only
        # with advVerified=False surfaced (never a silent liquidity pass).
        trad = tradability(price, account_jpy, adv_jpy=adv, max_round_trip_bps=max_round_trip_bps,
                           require_adv=adv is not None) if price else None
        if not (trad and trad["tradable"]):
            excluded["not_tradable"] += 1; continue
        actionable.append((ev, trad))

    per_h: dict[str, Any] = {}
    horizon_ns: list[int] = []
    for h in horizons:
        nets: list[float] = []
        n_missing = 0
        for ev, trad in actionable:
            raw = event_return_fn(ev, h)
            if raw is None:
                n_missing += 1   # unmatured / no return — ledgered per horizon, not silent
                continue
            # drift in the PREDICTED direction, net of the round-trip cost
            net = raw * ev["direction"] - trad["roundTripBps"] / 1e4
            nets.append(net)
        n = len(nets)
        horizon_ns.append(n)
        if nets:
            mean = sum(nets) / n
            std = (sum((x - mean) ** 2 for x in nets) / (n - 1)) ** 0.5 if n > 1 else 0.0
            per_h[str(h)] = {
                "n": n,
                "nMissing": n_missing,
                "meanNet": round(mean, 4),
                "hitRate": round(sum(1 for x in nets if x > 0) / n, 3),
                "perEventSharpe": round(mean / std, 3) if std > 0 else None,
            }
        else:
            per_h[str(h)] = {"n": 0, "nMissing": n_missing, "meanNet": None, "hitRate": None, "perEventSharpe": None}

    # Honest verdict — gated by the SLOWEST horizon (can't claim "evaluated" until the
    # longest-horizon events have matured), never by the most-mature single horizon.
    matured = min(horizon_ns) if horizon_ns else 0
    verdict = "insufficient_data" if matured < min_events else "evaluated"
    promotion = {
        "eligible": verdict == "evaluated",
        "note": ("promotion 须对搜索过的全部配置跑 overfit_gate.promote_gate(真实 cross-trial sr_std + skew/kurt),"
                 "再过 append-only forward log;本表只出描述统计,不替代准入闸(ADR-0010 / Codex)"),
    }
    return {
        "verdict": verdict,
        "minEvents": min_events,
        "eventsIn": len(events),
        "actionableCount": len(actionable),
        "excluded": excluded,
        "horizons": per_h,
        "promotion": promotion,
        "benchmarkNote": "净漂移已扣往返成本(按信号方向);未成熟/无回报按 horizon 计入 nMissing",
        "disclosure": DISCLOSURE,
    }
