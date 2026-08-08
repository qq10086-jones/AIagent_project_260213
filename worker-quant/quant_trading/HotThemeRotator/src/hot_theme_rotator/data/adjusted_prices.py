"""P35-01 — the one adjusted-return contract over raw ``daily_prices``.

Semantics, fixed here and nowhere else
---------------------------------------
- **raw price** — exactly what ``daily_prices`` stores: unadjusted closes
  (``auto_adjust=False``, Rule 11.9.6). Correct for latest tradable price,
  mark-to-market, entry/reference prices, OHLC display, volume/ADV/turnover.
- **adjusted price** — raw closes back-adjusted into the CURRENT share basis:
  prices before an N:1 split are divided by N. The latest adjusted price always
  equals the latest raw price.
- **adjusted return** — returns computed on adjusted prices. The ONLY
  research-grade multi-day return. 1306.T (the benchmark) carries a 10:1 split
  on 2026-03-30 that reads as −90.1% on raw prices; 63/2,774 symbols carry
  similar artifacts.

Fail-closed classification (hardened 2026-08-09)
-------------------------------------------------
A >45% one-day move is classified a split ONLY on positive evidence:

1. price ratio implies a near-integer factor (N:1 or 1:N, ±5%), AND
2. volume corroborates (share count multiplies ≈N×, wide band), OR the action
   is listed in ``verified_actions`` (an explicit, auditable override).

**A near-integer ratio with volume UNAVAILABLE is ``ambiguous``** — a plausible
ratio alone is one coincidence away from erasing a real crash, and silence is
not corroboration. Volume CONTRADICTION is likewise ``ambiguous``. Strict mode
raises on any ambiguous jump; non-strict leaves it unadjusted and reports it so
the caller can exclude the *intersecting windows* (not necessarily the whole
symbol — an anomaly outside every window a study reads contaminates nothing
that study computes; see :func:`ambiguous_indices`).

``verified_actions`` maps ISO date → factor (e.g. ``{"2026-03-30": 10.0}``) for
splits confirmed against an external source. It is passed per call, visible at
the call site, and never a ticker special-case buried here.

Input validation
----------------
Dates must be strictly increasing (no duplicates, no disorder), closes finite
and > 0, volumes finite and >= 0 when present. Violations raise — a corrupted
series must not flow silently into a return, and a bad denominator must never
become a fabricated 0% return.

Splits only; dividends are out of scope (price-return basis on both sides).
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

__all__ = [
    "CorporateActionError",
    "CorporateAction",
    "PriceBar",
    "validate_bars",
    "detect_corporate_actions",
    "adjust_prices",
    "adjusted_returns",
    "ambiguous_indices",
]

_JUMP_THRESHOLD = 0.45
_FACTOR_TOL = 0.05
_MAX_FACTOR = 100
_VOL_AGREE_LOW, _VOL_AGREE_HIGH = 0.2, 5.0


class CorporateActionError(ValueError):
    """Raised on unclassifiable jumps or invalid input series."""


@dataclass(frozen=True)
class PriceBar:
    date: str
    close: float
    volume: float | None = None


@dataclass(frozen=True)
class CorporateAction:
    index: int                 # bar index of the POST-action price
    date: str
    price_ratio: float         # post/pre
    factor: float              # N for N:1 (price ÷N); 1/N for reverse
    kind: str                  # "split" | "reverse_split" | "ambiguous"
    evidence: str              # "volume_agrees" | "verified_override" |
                               # "volume_unavailable" | "volume_contradicts" |
                               # "ratio_not_integer"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def validate_bars(bars: Sequence[PriceBar]) -> None:
    """Strictly increasing dates; finite closes > 0; sane volumes. Raises."""
    prev_date: str | None = None
    for i, b in enumerate(bars):
        if not isinstance(b.close, (int, float)) or isinstance(b.close, bool) \
                or not math.isfinite(b.close) or b.close <= 0:
            raise CorporateActionError(
                f"bar {i} ({b.date}): close must be finite and > 0, got {b.close!r} — "
                f"a bad denominator must never become a fabricated return")
        if b.volume is not None and (
                not isinstance(b.volume, (int, float)) or isinstance(b.volume, bool)
                or not math.isfinite(b.volume) or b.volume < 0):
            raise CorporateActionError(
                f"bar {i} ({b.date}): volume must be finite and >= 0, got {b.volume!r}")
        if prev_date is not None and b.date <= prev_date:
            reason = "duplicate" if b.date == prev_date else "out-of-order"
            raise CorporateActionError(
                f"bar {i}: {reason} date {b.date} after {prev_date}; a return "
                f"series over a disordered index is meaningless")
        prev_date = b.date


def _classify(pre: PriceBar, post: PriceBar,
              verified_factor: float | None) -> tuple[str, float, str]:
    """(kind, factor, evidence) for one jump."""
    ratio = post.close / pre.close
    if verified_factor is not None:
        kind = "split" if verified_factor > 1.0 else "reverse_split"
        return kind, float(verified_factor), "verified_override"

    implied = (1.0 / ratio) if ratio < 1.0 else ratio
    nearest = round(implied)
    if nearest < 2 or nearest > _MAX_FACTOR or abs(implied - nearest) / nearest > _FACTOR_TOL:
        return "ambiguous", implied, "ratio_not_integer"

    if not (pre.volume and post.volume and pre.volume > 0):
        # Plausible ratio, no volume: silence is not corroboration.
        return "ambiguous", float(nearest), "volume_unavailable"

    kind = "split" if ratio < 1.0 else "reverse_split"
    expected = float(nearest) if kind == "split" else 1.0 / nearest
    rel = (post.volume / pre.volume) / expected
    if not (_VOL_AGREE_LOW <= rel <= _VOL_AGREE_HIGH):
        return "ambiguous", float(nearest), "volume_contradicts"
    factor = float(nearest) if kind == "split" else 1.0 / nearest
    return kind, factor, "volume_agrees"


def detect_corporate_actions(
    bars: Sequence[PriceBar],
    *,
    threshold: float = _JUMP_THRESHOLD,
    verified_actions: Mapping[str, float] | None = None,
) -> list[CorporateAction]:
    """Scan a validated raw series for split-like jumps and classify each."""
    validate_bars(bars)
    verified = dict(verified_actions or {})
    out: list[CorporateAction] = []
    for i in range(len(bars) - 1):
        pre, post = bars[i], bars[i + 1]
        if abs(post.close / pre.close - 1.0) > threshold:
            kind, factor, evidence = _classify(pre, post, verified.get(post.date))
            out.append(CorporateAction(i + 1, post.date, post.close / pre.close,
                                       factor, kind, evidence))
    return out


def adjust_prices(
    bars: Sequence[PriceBar],
    *,
    strict: bool = True,
    verified_actions: Mapping[str, float] | None = None,
) -> tuple[list[float], list[CorporateAction]]:
    """Back-adjusted closes in the current share basis, plus what was adjusted.

    ``strict=True`` raises on any ambiguous jump. ``strict=False`` leaves
    ambiguous jumps UNADJUSTED and returns them — the caller must exclude the
    windows that intersect them (:func:`ambiguous_indices`); it never guesses.
    """
    actions = detect_corporate_actions(bars, verified_actions=verified_actions)
    ambiguous = [a for a in actions if a.kind == "ambiguous"]
    if ambiguous and strict:
        first = ambiguous[0]
        raise CorporateActionError(
            f"unclassifiable jump at {first.date} (ratio {first.price_ratio:.4f}, "
            f"evidence {first.evidence}): not a corroborated split. Refusing to "
            f"adjust — an invented factor would rewrite history. Supply "
            f"verified_actions={{'{first.date}': <factor>}} from an external "
            f"source, or pass strict=False and exclude intersecting windows.")

    adjusted = [b.close for b in bars]
    for action in actions:
        if action.kind == "ambiguous":
            continue
        for i in range(action.index):
            adjusted[i] /= action.factor
    return adjusted, actions


def ambiguous_indices(actions: Sequence[CorporateAction]) -> list[int]:
    """Bar indices of unresolved jumps. A window [a, b) is contaminated iff it
    contains one of these indices; windows elsewhere in the same symbol are
    fine — contamination is per-window, not per-symbol."""
    return [a.index for a in actions if a.kind == "ambiguous"]


def adjusted_returns(
    bars: Sequence[PriceBar],
    *,
    strict: bool = True,
    verified_actions: Mapping[str, float] | None = None,
) -> tuple[list[float], list[CorporateAction]]:
    """Split-adjusted simple returns — the only research-grade return series.

    Validation guarantees every denominator is finite and > 0, so no branch
    here can quietly emit a fabricated 0% return.
    """
    adjusted, actions = adjust_prices(bars, strict=strict,
                                      verified_actions=verified_actions)
    rets = [adjusted[i + 1] / adjusted[i] - 1.0 for i in range(len(adjusted) - 1)]
    return rets, actions
