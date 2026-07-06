"""SKHY event state + Japan semi sympathy overlay (P20-04 / Rule 11.15, 11.14).

Pure functions. SKHY is treated as an external *impulse*, not a forecast: we
measure amplitude (abnormal move), confirmation (SOX/peer/KR same-direction),
and freshness (half-life decay). The overlay may lightly annotate Japanese semi
candidates AFTER the existing rerank, but:

- a stale/pending SKHY leaves ranking unchanged (catalyst off);
- sector membership ALONE never earns a sympathy label (>=2 facts required);
- extended-chase names (Rule 11.14) get no boost (study-only);
- impact is capped tight ([-0.05, +0.07]) in normalized rerank space;
- NO probability / win-rate / expected-return / guaranteed field is produced.
"""
from __future__ import annotations

from datetime import date
from typing import Any, Mapping, Optional, Sequence

SEMI_THEME_TAGS = frozenset(
    {"semi", "semiconductor", "semiconductors", "memory", "ai_hardware", "ai hardware"}
)
DEFAULT_ABNORMAL_MOVE = 0.03      # |overnight move| that counts as an impulse
DEFAULT_HALF_LIFE_DAYS = 2.0      # catalyst freshness half-life

__all__ = ["compute_skhy_event", "annotate_japan_semi", "SEMI_THEME_TAGS"]


def _parse_date(value: Any) -> Optional[date]:
    if not value or not isinstance(value, str):
        return None
    try:
        return date.fromisoformat(value[:10])
    except ValueError:
        return None


def _freshness(asof: str, data_ts: Optional[str], half_life_days: float) -> float:
    a, d = _parse_date(asof), _parse_date(data_ts)
    if a is None or d is None:
        return 0.0
    age = (a - d).days
    if age < 0:
        return 0.0  # future-dated data → fail-closed, not fresh (P20 fix#3)
    if half_life_days <= 0:
        return 1.0 if age == 0 else 0.0
    return 0.5 ** (age / half_life_days)


def compute_skhy_event(
    adr: Mapping[str, Mapping[str, Any]],
    *,
    asof: str,
    abnormal_move: float = DEFAULT_ABNORMAL_MOVE,
    half_life_days: float = DEFAULT_HALF_LIFE_DAYS,
) -> dict[str, Any]:
    """Derive the SKHY catalyst event state from ADR-watch instrument snapshots.

    State machine: stale/pending SKHY -> off; active+abnormal but no confirmation
    -> watch; active+abnormal+confirmation -> active.
    """
    skhy = dict(adr.get("SKHY") or {})
    status = skhy.get("status", "unavailable")
    skhy_fresh = status == "active" and not skhy.get("stale", False)
    move = skhy.get("overnight_return")
    abnormal = move is not None and abs(move) >= abnormal_move

    # Confirmation must be INDEPENDENT: SOX (sector) or US memory/AI peers (MU/NVDA).
    # 000660.KS is SK hynix's own Korea line (same issuer as SKHY) — it is proxy/context,
    # NOT independent confirmation, so it is excluded here.
    confirms: list[str] = []
    if move is not None:
        for name, key in (("sox", "SOXX"), ("mu", "MU"), ("nvda", "NVDA")):
            d = adr.get(key) or {}
            r = d.get("overnight_return")
            if (
                d.get("status") == "active"
                and not d.get("stale", False)  # P20 fix#4 — a stale peer cannot confirm
                and r is not None
                and abs(r) >= abnormal_move * 0.5
                and (r > 0) == (move > 0)
            ):
                confirms.append(f"{name}_confirms")

    if not skhy_fresh or not abnormal:
        state, active = "off", False
    elif confirms:
        state, active = "active", True
    else:
        state, active = "watch", False

    freshness = _freshness(asof, skhy.get("data_ts"), half_life_days) if skhy_fresh else 0.0

    return {
        "skhyCatalystStatus": status,
        "skhyCatalystActive": active,
        "eventState": state,
        "skhyOvernightMove": move,
        "skhyFresh": bool(skhy_fresh),
        "confirmations": tuple(confirms),
        "abnormalMove": bool(abnormal),
        "freshness": round(freshness, 4),
        "disclosure": "External ADR catalyst; not a JP buy signal; no probability.",
    }


def _is_semi_member(c: Mapping[str, Any]) -> bool:
    if c.get("semiThemeMember") is True:
        return True
    themes = c.get("themes")
    if isinstance(themes, (list, tuple, set)):
        if any(str(t).lower() in SEMI_THEME_TAGS for t in themes):
            return True
    one = c.get("theme")
    return bool(one) and str(one).lower() in SEMI_THEME_TAGS


def _candidate_move(c: Mapping[str, Any]) -> Optional[float]:
    for k in ("recentReturn", "mom_5", "mom_20"):
        v = c.get(k)
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            return float(v)
    return None


def annotate_japan_semi(
    candidates: Sequence[Mapping[str, Any]],
    event: Mapping[str, Any],
    *,
    max_boost: float = 0.07,
    max_penalty: float = -0.05,
) -> list[dict[str, Any]]:
    """Annotate JP candidates with SKHY sympathy context (read-only, capped).

    A positive ``semiSympathyScore`` requires the catalyst to be fresh/on AND at
    least two facts (membership + active/watch confirmation, optionally relative
    strength). Membership alone, a stale catalyst, or an extended-chase flag all
    yield 0.
    """
    active = bool(event.get("skhyCatalystActive"))
    state = event.get("eventState", "off")
    fresh = bool(event.get("skhyFresh"))
    move = event.get("skhyOvernightMove")
    freshness = float(event.get("freshness") or 0.0)

    out: list[dict[str, Any]] = []
    for c in candidates:
        cc = dict(c)
        cc["skhyCatalystStatus"] = event.get("skhyCatalystStatus")
        cc["skhyCatalystActive"] = active
        cc["skhyOvernightMove"] = move
        cmove = _candidate_move(c)
        rel = (cmove - move) if (cmove is not None and move is not None) else None
        cc["relativeStrengthVsSkhy"] = round(rel, 4) if rel is not None else None

        member = _is_semi_member(c)
        # Rule 11.14 extended-chase: a real candidate carries chaseRisk as a string
        # ("none" / "study_only"), so only a non-"none" value (or leaderExtended) blocks the boost.
        extended = bool(c.get("leaderExtended")) or str(c.get("chaseRisk") or "").lower() not in ("", "none")
        score = 0.0
        if not member:
            reasons: list[str] = ["not_semi_theme_member"]
        elif state == "off" or not fresh:
            reasons = ["external_catalyst_off_or_stale"]
        elif extended:
            reasons = ["extended_chase_study_only_rule_11_14"]
        else:
            facts = ["semi_theme_member"]
            if active:
                facts.append("skhy_active_with_confirmation")
            elif state == "watch":
                facts.append("skhy_watch_unconfirmed")
            if rel is not None and rel >= 0:
                facts.append("relative_strength_holding")
            if len(facts) >= 2:
                base = max_boost if active else max_boost * 0.4
                score = round(min(base * freshness, max_boost), 4)
                reasons = facts
            else:
                reasons = facts + ["insufficient_facts"]

        cc["semiSympathyScore"] = max(max_penalty, min(max_boost, score))
        cc["semiSympathyReasons"] = tuple(reasons)
        out.append(cc)
    return out
