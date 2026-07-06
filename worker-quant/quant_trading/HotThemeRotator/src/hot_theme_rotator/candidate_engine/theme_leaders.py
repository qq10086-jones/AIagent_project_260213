"""Theme-leader annotation (ADR-0009 P15-02b).

Answers 00_DESIGN's core question "每个热点主题里真正的龙头是谁": within each hot
theme, the top blended-ranked (strength × news catalyst) candidate is that theme's
leader. Deterministic, derived from the already-ranked candidate list — no extra
data, no fabrication; an uncatalyzed candidate is never a theme leader.

(The full `leader_ranking.leader_ranker` within-theme scoring — volume expansion /
overheat penalty — is a deferred refinement: mapping the screener's factor schema to
its inputs needs deliberate tuning, tracked as P15-02b-full.)
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any


def annotate_theme_leaders(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Add `isThemeLeader` + `themeLeaderRank` to each candidate. Expects the list
    already sorted by blended rank (best first); the first catalyzed candidate per
    `topTheme` is rank 1 = the theme leader. Non-catalyzed candidates get rank None."""
    seen: dict[str, int] = defaultdict(int)
    out: list[dict[str, Any]] = []
    for c in candidates:
        theme = c.get("topTheme") if c.get("newsCatalyzed") else None
        rank = None
        if theme:
            seen[theme] += 1
            rank = seen[theme]
        out.append({**c, "isThemeLeader": rank == 1, "themeLeaderRank": rank})
    return out
