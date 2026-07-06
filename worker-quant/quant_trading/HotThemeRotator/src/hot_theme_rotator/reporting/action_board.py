"""Daily Action Board assembler (P21-03, Rule 11.16).

Joins already-served candidate fields — the blended ordering (Rule 11.12), the
7-tier price ladder (Rule 11.6/9.6), chase-risk / rotation annotations
(Rule 11.14) — with portfolio NAV/cash into read-only TRADE PLAN rows: entry
zone / stop reference / take-profit references / deterministic sizing
arithmetic (Rule 11.8) / factual timing conditions.

It introduces NO new predictive score (rows inherit the served order and may
only be downgraded), and by construction emits no probability, win-rate,
expected-return, or edge field (Rule 11.16.1-2). Sizing is arithmetic from an
explicit risk-budget fraction of NAV, capped by the Rule 12.5 concentration
limit and available cash; when NAV is unavailable sizing is ``None`` — never
estimated from a fabricated NAV (Rule 11.16.5).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

DISCLOSURE = (
    "交易计划 = 结构参考，不是预测；系统无 demonstrated edge；"
    "由你在外部券商手动决定与执行 — Rule 3 — manual execution outside HTR。"
    "本板不含概率/胜率/期望收益。"
)

_ENTRY_KINDS = {
    "entry_aggressive": "aggressive",
    "entry_balanced": "balanced",
    "entry_conservative": "conservative",
}
_EXIT_ORDER = ("exit_1", "exit_2", "exit_stretch")


@dataclass(frozen=True)
class BoardConfig:
    risk_budget_frac: float = 0.01        # Rule 11.8: explicit risk-budget % of NAV
    concentration_cap_frac: float = 0.20  # Rule 12.5 concentration limit
    lot_size: int = 100                   # JPX whole-lot unit
    top_n: int = 8


def build_action_board(
    candidates: list[dict[str, Any]],
    *,
    nav_jpy: float | None,
    cash_jpy: float | None,
    market_session: str | None = None,
    config: BoardConfig = BoardConfig(),
) -> dict[str, Any]:
    """Assemble the board from served candidate dicts (fail-open per row)."""
    rows = []
    for candidate in list(candidates or [])[: int(config.top_n)]:
        try:
            rows.append(_build_row(candidate, nav_jpy, cash_jpy, market_session, config))
        except Exception:  # noqa: BLE001 — one bad row never breaks the board
            rows.append(_identity(candidate) | {
                "planStatus": "insufficient_data",
                "planStatusReasons": ["row_assembly_error"],
                "entryPlan": None, "stopRef": None, "exitRefs": [],
                "sizing": None, "timingChecklist": [], "conditionsMet": "0/0",
            })
    return {
        "disclosure": DISCLOSURE,
        "scoreStatus": "uncalibrated_research_score",
        "marketSession": market_session,
        "navJpy": nav_jpy,
        "cashJpy": cash_jpy,
        "params": {
            "riskBudgetFrac": config.risk_budget_frac,
            "concentrationCapFrac": config.concentration_cap_frac,
            "lotSize": config.lot_size,
        },
        "rows": rows,
    }


def _identity(candidate: dict[str, Any]) -> dict[str, Any]:
    return {
        "symbol": candidate.get("symbol"),
        "nameJa": candidate.get("nameJa") or "",
        "price": candidate.get("price"),
        "topTheme": candidate.get("topTheme"),
        "newsCatalyzed": bool(candidate.get("newsCatalyzed")),
        "catalystEvidenceLevel": candidate.get("catalystEvidenceLevel", "none"),
        "chaseRisk": candidate.get("chaseRisk", "none"),
        "leaderExtended": bool(candidate.get("leaderExtended")),
        "isThemeLeader": bool(candidate.get("isThemeLeader")),
        "rotationReasons": list(candidate.get("rotationReasons", []) or []),
        "why": candidate.get("reason") or candidate.get("one_liner") or "",
    }


def _tiers_by_kind(candidate: dict[str, Any]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for tier in candidate.get("ladder", []) or []:
        kind = tier.get("kind")
        if kind and tier.get("price") is not None:
            out[kind] = {"label": tier.get("label"), "price": float(tier["price"]),
                         "pct": tier.get("pct")}
    return out


def _build_row(
    candidate: dict[str, Any],
    nav_jpy: float | None,
    cash_jpy: float | None,
    market_session: str | None,
    config: BoardConfig,
) -> dict[str, Any]:
    row = _identity(candidate)
    tiers = _tiers_by_kind(candidate)
    price = float(candidate.get("price") or 0.0)
    entry = tiers.get("entry_balanced", {}).get("price")
    stop = tiers.get("stop", {}).get("price")

    reasons: list[str] = []
    # --- fail-closed data gate (Rule 11.16.3) -------------------------------
    if price <= 0 or entry is None or stop is None or entry <= stop:
        row.update({
            "planStatus": "insufficient_data",
            "planStatusReasons": ["missing_price_or_ladder" if entry is None or stop is None or price <= 0
                                  else "stop_not_below_entry"],
            "entryPlan": None, "stopRef": None, "exitRefs": [],
            "sizing": None,
            "timingChecklist": _timing(candidate, market_session),
        })
        row["conditionsMet"] = _met(row["timingChecklist"])
        return row

    row["entryPlan"] = {
        name: tiers[kind] for kind, name in _ENTRY_KINDS.items() if kind in tiers
    }
    row["stopRef"] = tiers["stop"]
    row["exitRefs"] = [tiers[k] for k in _EXIT_ORDER if k in tiers]

    # --- deterministic sizing arithmetic (Rule 11.8 / 11.16.5) --------------
    sizing = None
    shares = None
    whole_lot_shares = None
    if nav_jpy and nav_jpy > 0 and cash_jpy is not None:
        risk_budget_jpy = nav_jpy * config.risk_budget_frac
        risk_per_share = entry - stop
        shares_by_risk = math.floor(risk_budget_jpy / risk_per_share)
        cap_jpy = min(float(cash_jpy), nav_jpy * config.concentration_cap_frac)
        shares_by_cap = math.floor(cap_jpy / entry) if entry > 0 else 0
        shares = max(0, min(shares_by_risk, shares_by_cap))
        whole_lots = shares // config.lot_size
        whole_lot_shares = whole_lots * config.lot_size
        sizing = {
            "method": "risk_budget_arithmetic",  # Rule 11.8 — 算术测算，非指令
            "riskBudgetFrac": config.risk_budget_frac,
            "riskBudgetJpy": round(risk_budget_jpy, 0),
            "riskPerShare": round(risk_per_share, 2),
            "sharesAtRiskBudget": shares,
            "sKabuShares": shares,
            "wholeLots": whole_lots,
            "wholeLotShares": whole_lot_shares,
            "notionalWholeLotJpy": round(whole_lot_shares * entry, 0),
            "capJpy": round(cap_jpy, 0),
            "concentrationCapFrac": config.concentration_cap_frac,
        }
    row["sizing"] = sizing

    # --- planStatus resolution, fail-closed (Rule 11.16.3) ------------------
    chase = str(candidate.get("chaseRisk", "none") or "none")
    if chase == "study_only" or bool(candidate.get("leaderExtended")):
        status = "watch_only"
        reasons.append("chase_study_only" if chase == "study_only" else "leader_extended")
    elif sizing is not None and shares is not None and shares < 1:
        status = "not_tradable"
        reasons.append("no_affordable_shares_within_caps")
    elif sizing is not None and whole_lot_shares is not None and whole_lot_shares < config.lot_size:
        status = "s_kabu_only"
        reasons.append("whole_lot_exceeds_caps_s_kabu_feasible")
    else:
        status = "plan_ready"
        if sizing is None:
            # governance review 2026-07-06 nit#3: without NAV the tradability
            # arithmetic never ran — say so instead of implying it passed.
            reasons.append("tradability_unverified_no_nav")
    row["planStatus"] = status
    row["planStatusReasons"] = reasons

    row["timingChecklist"] = _timing(candidate, market_session)
    row["conditionsMet"] = _met(row["timingChecklist"])
    return row


def _timing(candidate: dict[str, Any], market_session: str | None) -> list[dict[str, Any]]:
    """Factual conditions only — never a clock call or urgency cue (Rule 11.16.6)."""
    chase = str(candidate.get("chaseRisk", "none") or "none")
    evidence = str(candidate.get("catalystEvidenceLevel", "none") or "none")
    return [
        {"key": "market_session", "label": "市场时段", "value": market_session or "UNKNOWN",
         "ok": market_session == "OPEN"},
        {"key": "chase_risk", "label": "追高风险", "value": chase, "ok": chase == "none"},
        {"key": "catalyst_evidence", "label": "催化证据", "value": evidence,
         "ok": evidence == "company"},
        {"key": "leader_extended", "label": "龙头过热", "value": bool(candidate.get("leaderExtended")),
         "ok": not bool(candidate.get("leaderExtended"))},
    ]


def _met(checklist: list[dict[str, Any]]) -> str:
    return f"{sum(1 for c in checklist if c.get('ok'))}/{len(checklist)}"
