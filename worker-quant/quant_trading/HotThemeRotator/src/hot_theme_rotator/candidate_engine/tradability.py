"""Execution / tradability gate (ADR-0010, Rule 5.1 — Codex execution-first mandate).

Phase 0 (2026-06-17) found the inferred small-cap disclosure-drift edge is tradable
only in a NARROW window for a ~¥400k retail account: the cheapest names (lot-size
fits) carry the worst JPX tick cost, while expensive names can't be diversified at
100-share lots. This module turns that into a deterministic, fail-closed gate.

Everything here is verifiable arithmetic (tick ladder, lot math, net-of-cost) — no
prediction, no probability. A signal whose expected gross move does not clear the
round-trip cost (and a 2x-cost stress) inside the affordable window is NOT actionable.
"""
from __future__ import annotations

from typing import Any

LOT = 100  # JPX domestic standard trading unit (shares per lot)

# JPX tick ladder for non-TOPIX500 "Other Issues" (the small/illiquid names this
# strategy targets). Ascending (upper_price_inclusive, tick_yen); last entry catches
# the tail. Matches the regime Codex cited (¥1 tick up to ¥3,000).
_TICK_LADDER: tuple[tuple[float, float], ...] = (
    (3_000, 1.0),
    (5_000, 5.0),
    (30_000, 10.0),
    (50_000, 50.0),
    (300_000, 100.0),
    (float("inf"), 1_000.0),
)


def jpx_tick(price: float) -> float:
    """Minimum price increment (¥) for a non-TOPIX500 issue at `price`."""
    for upper, tick in _TICK_LADDER:
        if price <= upper:
            return tick
    return 1_000.0  # unreachable (inf guard), kept explicit


def round_trip_cost_bps(price: float, *, spread_ticks: float = 3.0, slippage_bps: float = 0.0) -> float:
    """Estimated round-trip cost in bps from the tick-implied spread.

    `spread_ticks` = total ticks paid across entry+exit. Default 3 (NOT 2) per Codex
    review — thin small-caps rarely sit 1-tick-wide, so 2 flattered the gate; widen to
    4-5 for very illiquid names. Liquidity/impact slippage is added on top. Honest
    lower-bound: real illiquid spreads are often wider still."""
    if spread_ticks < 0 or slippage_bps < 0:
        raise ValueError("spread_ticks and slippage_bps must be non-negative")
    if price <= 0:
        return float("inf")
    tick = jpx_tick(price)
    return spread_ticks * (tick / price) * 1e4 + slippage_bps


def lots_affordable(price: float, account_jpy: float, *, max_position_frac: float = 0.34) -> dict[str, Any]:
    """How many 100-share lots fit, capped so no single name exceeds
    `max_position_frac` of the account (diversification guard for a small account)."""
    lot_cost = price * LOT
    if lot_cost <= 0 or account_jpy <= 0:
        return {"lotCost": lot_cost, "lots": 0, "positionFrac": float("inf"), "diversifiable": False}
    cap_lots = int((account_jpy * max_position_frac) // lot_cost)
    return {
        "lotCost": round(lot_cost, 0),
        "lots": max(cap_lots, 0),
        "positionFrac": round(lot_cost / account_jpy, 4),
        # diversifiable = can hold >=3 distinct single-lot names without breaching cap
        "diversifiable": lot_cost <= account_jpy * max_position_frac,
    }


def net_expected_after_cost(
    gross_return: float, price: float, *, spread_ticks: float = 3.0, slippage_bps: float = 10.0
) -> float:
    """Expected return net of the round-trip cost fraction (gross is a decimal, e.g.
    0.03 = 3%). Slippage defaults to a non-zero retail estimate so we never flatter."""
    return gross_return - round_trip_cost_bps(price, spread_ticks=spread_ticks, slippage_bps=slippage_bps) / 1e4


def tradability(
    price: float,
    account_jpy: float = 400_000.0,
    *,
    adv_jpy: float | None = None,
    expected_gross: float | None = None,
    min_adv_jpy: float = 50_000_000.0,
    max_round_trip_bps: float = 60.0,
    spread_ticks: float = 3.0,
    slippage_bps: float = 10.0,
    max_position_frac: float = 0.34,
    require_adv: bool = True,
) -> dict[str, Any]:
    """Fail-closed tradability verdict for one name at `price` for `account_jpy`.

    `tradable` requires ALL of: at least one lot affordable within the position cap,
    round-trip cost ≤ `max_round_trip_bps`, the liquidity gate (see below), and — when
    an `expected_gross` move is supplied — that it survives 2× the round-trip cost.
    `reasons` lists every failing gate (never silently drops a constraint).

    Liquidity gate (Codex fix): ADV is fail-closed. When `adv_jpy` is None and
    `require_adv` is True, the name fails with an "ADV unverified" reason rather than
    silently passing. Callers that genuinely lack volume data (e.g. the Event Desk
    priced-in read) pass `require_adv=False` to get the structural cost+lot verdict
    with `advVerified=False` surfaced — never a silent liquidity pass."""
    if not isinstance(price, (int, float)) or price <= 0:
        return {"price": price, "tick": None, "roundTripBps": None, "lotCost": None,
                "lotsAffordable": 0, "positionFrac": None, "advVerified": False, "advOk": False,
                "netAfterCost": None, "netVerified": False, "survives2xCost": None,
                "tradable": False, "reasons": ["invalid price (<= 0 or non-numeric)"]}

    rt_bps = round_trip_cost_bps(price, spread_ticks=spread_ticks, slippage_bps=slippage_bps)
    lots = lots_affordable(price, account_jpy, max_position_frac=max_position_frac)
    reasons: list[str] = []

    if lots["lots"] < 1 or not lots["diversifiable"]:
        reasons.append(f"lot ¥{lots['lotCost']:.0f} = {lots['positionFrac']*100:.0f}% of account (>{max_position_frac*100:.0f}% cap / can't diversify)")
    if rt_bps > max_round_trip_bps:
        reasons.append(f"round-trip {rt_bps:.0f}bps > {max_round_trip_bps:.0f}bps cap")

    # Liquidity gate — fail-closed (Codex): unverified ADV does not silently pass.
    adv_verified = adv_jpy is not None
    adv_ok = True
    if adv_verified:
        adv_ok = adv_jpy >= min_adv_jpy
        if not adv_ok:
            reasons.append(f"ADV ¥{adv_jpy/1e6:.0f}M < ¥{min_adv_jpy/1e6:.0f}M floor")
    elif require_adv:
        adv_ok = False
        reasons.append("ADV unverified (no volume data) — liquidity gate fails closed")

    net = None
    survives_2x = None
    if expected_gross is not None:
        net = round(net_expected_after_cost(expected_gross, price, spread_ticks=spread_ticks, slippage_bps=slippage_bps), 4)
        # 2x-cost stress: gross must still beat double the round-trip cost.
        survives_2x = expected_gross > 2.0 * (rt_bps / 1e4)
        if not survives_2x:
            reasons.append(f"gross {expected_gross*100:.1f}% fails 2x-cost stress ({2*rt_bps:.0f}bps)")

    return {
        "price": price,
        "tick": jpx_tick(price),
        "roundTripBps": round(rt_bps, 1),
        "lotCost": lots["lotCost"],
        "lotsAffordable": lots["lots"],
        "positionFrac": lots["positionFrac"],
        "advVerified": adv_verified,
        "advOk": adv_ok,
        "netAfterCost": net,
        "netVerified": expected_gross is not None,
        "survives2xCost": survives_2x,
        "tradable": len(reasons) == 0,
        "reasons": reasons,
    }


# ── S株 (fractional / 単元未満株) execution mode — Rule 5.2 (added 2026-06-24) ──
#
# Lot mode above assumes 100-share lots, which structurally excludes expensive
# names for a small account (a ¥77,600 name needs ¥7.76M/lot). S株 makes the min
# unit 1 share, so such names become tradable; and SBI ゼロ革命 makes S株 buy AND
# sell commission-free AND spread-free — the only cost is a session-reference
# timing buffer (S株 fills 3x/day at 始値/後場始値/大引け, no intraday timing, no
# limit orders → suits horizons >=5D, Rule 16.6). Sources verified 2026-06-24
# (SBI ゼロ革命 / S株 取引ルール).

S_KABU_TIMING_BUFFER_BPS = 5.0  # honest non-zero default; not a commission/spread


def s_kabu_round_trip_cost_bps(*, timing_buffer_bps: float = S_KABU_TIMING_BUFFER_BPS) -> float:
    """S株 round-trip cost in bps. Commission=0, spread=0 (SBI ゼロ革命); the
    only modelled cost is a session-reference timing buffer (price-independent)."""
    if timing_buffer_bps < 0:
        raise ValueError("timing_buffer_bps must be non-negative")
    return float(timing_buffer_bps)


def s_kabu_affordable(
    price: float,
    account_jpy: float,
    *,
    max_position_frac: float = 0.34,
    concentration_warn_frac: float = 0.20,
) -> dict[str, Any]:
    """Affordability under S株: min unit = 1 share. A name is affordable if a
    single share fits the position cap; ``sharesAffordable`` is how many shares
    fit. ``concentrationWarn`` flags when even 1 share exceeds the Rule 12.5
    20%-NAV concentration warning band (a warning, not a hard block)."""
    if price <= 0 or account_jpy <= 0:
        return {"unitCost": price, "sharesAffordable": 0, "positionFrac": float("inf"),
                "affordable": False, "concentrationWarn": True}
    frac_one = price / account_jpy
    max_shares = int((account_jpy * max_position_frac) // price)
    return {
        "unitCost": round(price, 0),
        "sharesAffordable": max(max_shares, 0),
        "positionFrac": round(frac_one, 4),  # fraction of account for ONE share
        "affordable": price <= account_jpy * max_position_frac,
        "concentrationWarn": frac_one > concentration_warn_frac,
    }


def s_kabu_tradability(
    price: float,
    account_jpy: float = 400_000.0,
    *,
    adv_jpy: float | None = None,
    expected_gross: float | None = None,
    min_adv_jpy: float = 50_000_000.0,
    timing_buffer_bps: float = S_KABU_TIMING_BUFFER_BPS,
    max_position_frac: float = 0.34,
    concentration_warn_frac: float = 0.20,
    require_adv: bool = True,
) -> dict[str, Any]:
    """Fail-closed S株-mode tradability verdict (Rule 5.2). Mirrors
    :func:`tradability` but with 1-share affordability and the S株 cost model.
    Note the timing caveat: S株 fills 3x/day at session reference prices, so it
    is for >=5D horizons, not intraday signals."""
    if not isinstance(price, (int, float)) or price <= 0:
        return {"price": price, "execMode": "s_kabu", "roundTripBps": None, "unitCost": None,
                "sharesAffordable": 0, "positionFrac": None, "advVerified": False, "advOk": False,
                "concentrationWarn": True, "netAfterCost": None, "netVerified": False,
                "survives2xCost": None, "tradable": False,
                "reasons": ["invalid price (<= 0 or non-numeric)"]}

    rt_bps = s_kabu_round_trip_cost_bps(timing_buffer_bps=timing_buffer_bps)
    aff = s_kabu_affordable(price, account_jpy, max_position_frac=max_position_frac,
                            concentration_warn_frac=concentration_warn_frac)
    reasons: list[str] = []
    if not aff["affordable"]:
        reasons.append(
            f"1 share ¥{aff['unitCost']:.0f} = {aff['positionFrac']*100:.0f}% of account "
            f"(>{max_position_frac*100:.0f}% cap)"
        )

    adv_verified = adv_jpy is not None
    adv_ok = True
    if adv_verified:
        adv_ok = adv_jpy >= min_adv_jpy
        if not adv_ok:
            reasons.append(f"ADV ¥{adv_jpy/1e6:.0f}M < ¥{min_adv_jpy/1e6:.0f}M floor")
    elif require_adv:
        adv_ok = False
        reasons.append("ADV unverified (no volume data) — liquidity gate fails closed")

    net = None
    survives_2x = None
    if expected_gross is not None:
        net = round(expected_gross - rt_bps / 1e4, 4)
        survives_2x = expected_gross > 2.0 * (rt_bps / 1e4)
        if not survives_2x:
            reasons.append(f"gross {expected_gross*100:.1f}% fails 2x-cost stress ({2*rt_bps:.0f}bps)")

    return {
        "price": price,
        "execMode": "s_kabu",
        "roundTripBps": round(rt_bps, 1),
        "unitCost": aff["unitCost"],
        "sharesAffordable": aff["sharesAffordable"],
        "positionFrac": aff["positionFrac"],
        "advVerified": adv_verified,
        "advOk": adv_ok,
        "concentrationWarn": aff["concentrationWarn"],
        "netAfterCost": net,
        "netVerified": expected_gross is not None,
        "survives2xCost": survives_2x,
        "tradable": len(reasons) == 0,
        "reasons": reasons,
    }
