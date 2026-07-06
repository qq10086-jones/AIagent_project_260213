"""S株 universe overlay — bring held/watchlisted expensive names into scope (Rule 5.2).

Selection (the top-N factor screen) lives in the sibling screener and is READ-ONLY
(ADR-0005); this overlay does NOT change selection. It brings the names the user
actually HOLDS (journal) or WATCHLISTS (user_state — HTR-owned, Rule 14.9) into the
candidate universe assessed in S株 mode, so the system is no longer blind to
expensive names (e.g. 8035.T) that the 100-share-lot gate excludes. It surfaces only
names that S株 *unlocks* (lot-untradable but S株-tradable), tagged
``execMode='s_kabu'``, ``source='s_kabu_overlay'``.

No edge is implied — these are held/tracked names made visible and cost-assessed,
NOT screener alpha picks. The Rule 12.5 concentration warning rides along so a name
that is one S株 share but >20% of NAV is shown with that caveat.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Optional, Sequence

from hot_theme_rotator.candidate_engine.tradability import s_kabu_tradability, tradability

__all__ = [
    "PriceLookup",
    "s_kabu_overlay_rows",
    "held_and_watchlist_names",
    "kline_price_lookup",
    "build_s_kabu_overlay",
]

PriceLookup = Callable[[str], Optional[float]]


def s_kabu_overlay_rows(
    names: Sequence[str],
    *,
    account_jpy: float,
    price_lookup: PriceLookup,
    adv_lookup: Optional[Callable[[str], Optional[float]]] = None,
    only_lot_unaffordable: bool = True,
) -> list[dict[str, Any]]:
    """Assess each name in S株 mode; return candidate rows S株 unlocks.

    Pure (lookups injected → unit-testable). By default (``only_lot_unaffordable``)
    a name is included only if the lot gate would have DROPPED it (lot-untradable)
    but S株 makes it tradable — i.e. exactly the expensive names this overlay exists
    for. Names without a usable price, or not S株-tradable (1 share > position cap),
    are skipped.
    """
    rows: list[dict[str, Any]] = []
    for sym in names:
        px = price_lookup(sym)
        if px is None or px <= 0:
            continue
        adv = adv_lookup(sym) if adv_lookup else None
        require_adv = adv is not None
        sk = s_kabu_tradability(px, account_jpy, adv_jpy=adv, require_adv=require_adv)
        lot = tradability(px, account_jpy, adv_jpy=adv, require_adv=require_adv)
        if only_lot_unaffordable and lot["tradable"]:
            continue  # lot mode already handles it; overlay is only for S株-unlocked names
        if not sk["tradable"]:
            continue  # 1 share still breaches the position cap → S株 can't help
        rows.append(
            {
                "symbol": sym,
                "price": px,
                "execMode": "s_kabu",
                "source": "s_kabu_overlay",
                "sharesAffordable": sk["sharesAffordable"],
                "positionFrac": sk["positionFrac"],
                "concentrationWarn": sk["concentrationWarn"],
                "roundTripBps": sk["roundTripBps"],
                "lotTradable": lot["tradable"],
                "advVerified": sk["advVerified"],
                "reasons": sk["reasons"],
            }
        )
    return rows


def held_and_watchlist_names(base_dir: str | Path = ".") -> list[str]:
    """Names the user holds (journal positions qty>0) ∪ watchlists (user_state).

    Both sources are HTR-owned (ADR-0005-safe). Defensive: a failure in either
    source contributes no names rather than raising.
    """
    names: set[str] = set()
    try:
        from hot_theme_rotator.portfolio.derive import derive_positions
        from hot_theme_rotator.portfolio.journal_writer import read_all_journal

        pos = derive_positions(read_all_journal(base_dir))
        names |= {sym for sym, v in pos.items() if getattr(v, "qty", 0) > 0}
    except Exception:
        pass
    try:
        from hot_theme_rotator.user_state.watchlist import load_watchlist

        state = load_watchlist(base_dir=base_dir)
        for e in getattr(state, "entries", ()) or ():
            sym = getattr(e, "symbol", None) or (e.get("symbol") if isinstance(e, dict) else None)
            if sym:
                names.add(str(sym))
    except Exception:
        pass
    return sorted(names)


def kline_price_lookup(db_path=None) -> PriceLookup:
    """Latest-close price lookup via the kline DB (None on any miss)."""
    from hot_theme_rotator.data.kline_adapter import default_db_path, fetch_latest_close

    path = db_path or default_db_path()

    def lookup(symbol: str) -> Optional[float]:
        try:
            bar = fetch_latest_close(path, symbol=symbol)
            close = getattr(bar, "close", None) if bar is not None else None
            return float(close) if close not in (None, 0) else None
        except Exception:
            return None

    return lookup


def build_s_kabu_overlay(
    base_dir: str | Path = ".",
    *,
    account_jpy: float = 400_000.0,
    price_lookup: Optional[PriceLookup] = None,
) -> dict[str, Any]:
    """Compose the overlay for held+watchlisted names using the kline price lookup.

    Returns a snapshot-shaped dict; ``candidates`` are the S株-unlocked rows.
    """
    names = held_and_watchlist_names(base_dir)
    lookup = price_lookup or kline_price_lookup()
    rows = s_kabu_overlay_rows(names, account_jpy=account_jpy, price_lookup=lookup)
    return {
        "execMode": "s_kabu",
        "account_jpy": account_jpy,
        "source": "s_kabu_overlay",
        "names_considered": names,
        "candidates": rows,
    }
