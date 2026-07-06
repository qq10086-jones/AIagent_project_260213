"""Disclosure-surprise + novelty signal (ADR-0010 P17-2).

Turns a TDnet disclosure title into the honest, parseable proxies for a PEAD-style
event signal:
  1. materiality — earnings / guidance-revision / dividend vs noise;
  2. surprise DIRECTION from the title's own words (上方修正 / 増配 = +1;
     下方修正 / 減配 / 無配 / 赤字 = -1; else 0);
  3. novelty — a genuinely new disclosure vs a 訂正 (correction) / reprint. Tetlock
     (2011): stale-news spikes REVERSE, so low novelty is a fade-not-chase flag.

It does NOT predict the drift or attach a probability (Rule 9.4 / 8.3). It flags which
disclosures are candidate events and in which direction; whether drift actually follows
is decided by the append-only forward log, gated by `tradability` (Rule 5.1) and the
anti-overfit promotion gate (ADR-0010 P17-3) before any "edge" is ever claimed.
"""
from __future__ import annotations

import re
from typing import Any

from hot_theme_rotator.data.external.tdnet_parser import classify_category

# Material disclosure categories (the PEAD-bearing ones). Guidance revisions classify
# as "earnings" under the parser's 業績/通期予想 rule.
_MATERIAL_CATEGORIES = frozenset({"earnings", "dividend"})

# Surprise direction from the title's own wording — these are statements of fact in
# the disclosure, not a forecast by us.
_UP = re.compile(r"(上方修正|上振れ|増配|復配|黒字転換|過去最高)")
_DOWN = re.compile(r"(下方修正|下振れ|減配|無配|赤字転落|業績悪化)")
# Staleness / non-novelty: corrections, re-releases, partial amendments.
_STALE = re.compile(r"(訂正|一部訂正|再送|再開示|の追加)")


def surprise_signal(title: str, *, category: str | None = None) -> dict[str, Any]:
    """Classify one disclosure title into {category, direction, material, novel,
    surpriseScore}. `surpriseScore` ∈ [-1, 1]: direction dampened when the disclosure
    is immaterial (×0.3) or stale/amended (×0.2). 0 when no clear surprise word."""
    if not isinstance(title, str) or not title.strip():
        return {"category": "other", "direction": 0, "mixed": False, "material": False, "novel": True, "surpriseScore": 0.0}
    # Reclassify from the title unless the caller supplied a SPECIFIC (non-"other")
    # category — "other" must not block a clear title match (Codex fix).
    cat = category if (category and category != "other") else classify_category(title)
    material = cat in _MATERIAL_CATEGORIES
    novel = not bool(_STALE.search(title))
    up = bool(_UP.search(title))
    down = bool(_DOWN.search(title))
    # Mixed signal (both up- and down-words) is ambiguous → direction 0, flagged.
    # Never let "up" silently win a contradictory title (Codex fix).
    mixed = up and down
    direction = 0 if mixed else (1 if up else (-1 if down else 0))
    mat_w = 1.0 if material else 0.3
    nov_w = 1.0 if novel else 0.2
    return {
        "category": cat,
        "direction": direction,
        "mixed": mixed,
        "material": material,
        "novel": novel,
        "surpriseScore": round(direction * mat_w * nov_w, 3),
    }


def rank_disclosures(disclosures: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Annotate each disclosure (dict with a 'title') with its surprise signal and
    return them sorted by |surpriseScore| desc (strongest, freshest, most material
    events first). Non-events (score 0) sink to the bottom; nothing is dropped."""
    out: list[dict[str, Any]] = []
    for d in disclosures:
        if not isinstance(d, dict):  # tolerate stray non-dict entries (Codex fix)
            continue
        sig = surprise_signal(d.get("title", ""), category=d.get("category"))
        # PIT guard: event-time use requires the Japanese release timestamp; flag rows
        # without one so the P17-4 validation harness can refuse look-ahead-biased
        # events rather than scoring them blind (Codex fix / ADR-0010 timestamp gate).
        pit_ok = bool(d.get("published_ts"))
        out.append({**d, **sig, "pitOk": pit_ok})
    out.sort(key=lambda r: abs(r.get("surpriseScore", 0.0)), reverse=True)
    return out
