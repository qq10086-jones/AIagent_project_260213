"""Owner Risk Mandate sleeve engine (P25-03, Section 17 / ADR-0012).

Read-only assembler between the journal-derived portfolio state (Section 14
SSoT, via the serialized positions dict) and the OWNER-DECLARED risk mandate
in ``configs/risk_mandate.json``: per-sleeve capital and β-adjusted exposure,
total exposure ratio vs the declared band (Rule 17.2), kill-switch buffer
(Rule 17.3), and Sleeve C discipline flags (Rule 17.4 — thesis_missing /
cap_breached / review_required). It predicts nothing, never acts (Rule 3),
carries per-sleeve honest expectation labels (Rule 17.6), and:

- fail-opens to ``None`` when the mandate config or portfolio is unavailable
  (never fabricated state, Rule 11.9.4);
- fail-closes unmapped holdings into an ``UNASSIGNED`` bucket with a warning
  (Rule 17.1 — never silently attributed to a sleeve).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

DISCLOSURE = (
    "风险授权面板 = 你自己声明的实验资本风险预算（Section 17 / ADR-0012）的算术对照，"
    "不是预测、不是下单指令；系统无 demonstrated edge；接受更高风险只放大方差，不创造 edge；"
    "操作在外部券商完成后经手工记录入账 — Rule 3 — manual execution outside HTR。"
    "不含概率/胜率/期望收益。"
)

MANDATE_RELPATH = Path("configs") / "risk_mandate.json"

# Sleeve C flags (Rule 17.4) — ordered by severity for display.
FLAG_THESIS_MISSING = "thesis_missing"
FLAG_CAP_BREACHED = "cap_breached"
FLAG_REVIEW_REQUIRED = "review_required"
FLAG_EXIT_TRIGGERED = "exit_triggered"  # Rule 17.4.6 — bilateral exit bracket touched


def _sector_look_through(
    holding_rows: list[dict[str, Any]], nav: float, mandate: dict[str, Any]
) -> list[dict[str, Any]]:
    """Rule 17.7 — penetrated sector exposure.

    Combines DIRECT theme tags (``theme_map``) with the sector weight EMBEDDED
    inside broad-index ETFs (``benchmark_sector_weights`` × leverage_factor).
    Observability only: it predicts nothing and issues no advice — it exists so
    a concentration hidden inside a passive holding cannot stay invisible.
    """
    theme_map = dict(mandate.get("theme_map") or {})
    bench_weights = dict(mandate.get("benchmark_sector_weights") or {})
    lev = dict(mandate.get("leverage_factors") or {})

    def _lev(sym: str) -> float:
        try:
            return float(lev.get(sym, lev.get("_default", 1.0)))
        except (TypeError, ValueError):
            return 1.0

    themes: dict[str, dict[str, float]] = {}
    for r in holding_rows:
        sym = r.get("symbol")
        if not sym or sym.startswith("_"):
            continue
        mv = float(r.get("marketValueJpy") or 0.0)
        theme = theme_map.get(sym)
        if theme:
            themes.setdefault(theme, {"directJpy": 0.0, "viaBenchmarkJpy": 0.0})["directJpy"] += mv
        weights = bench_weights.get(sym)
        if isinstance(weights, dict):
            for th, w in weights.items():
                if th.startswith("_"):
                    continue
                try:
                    wv = float(w)
                except (TypeError, ValueError):
                    continue
                themes.setdefault(th, {"directJpy": 0.0, "viaBenchmarkJpy": 0.0})[
                    "viaBenchmarkJpy"] += mv * _lev(sym) * wv

    out: list[dict[str, Any]] = []
    for th, v in themes.items():
        total = v["directJpy"] + v["viaBenchmarkJpy"]
        out.append({
            "theme": th,
            "directJpy": round(v["directJpy"], 2),
            "viaBenchmarkJpy": round(v["viaBenchmarkJpy"], 2),
            "totalJpy": round(total, 2),
            "fracNav": round(total / nav, 4) if nav else None,
        })
    out.sort(key=lambda x: x["totalJpy"], reverse=True)
    return out


def load_mandate(base_dir: str | Path) -> dict[str, Any] | None:
    """Load the owner-declared mandate; ``None`` (fail-open) when absent/invalid."""
    path = Path(base_dir) / MANDATE_RELPATH
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    # Minimal shape check — a mandate without these is not a mandate (fail-open).
    required = ("kill_switch_nav_floor_jpy", "target_exposure_ratio", "exposure_band", "sleeves", "sleeve_map")
    if not isinstance(raw, dict) or any(k not in raw for k in required):
        return None
    return raw


def build_risk_mandate_panel(
    positions: dict[str, Any] | None,
    *,
    base_dir: str | Path,
    mandate: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Assemble the Section 17 risk-mandate panel.

    Returns ``None`` (fail-open) when either the mandate config or the
    portfolio state is unavailable — the frontend then renders nothing.
    """
    if mandate is None:
        mandate = load_mandate(base_dir)
    if not mandate:
        return None
    if not positions or not positions.get("available"):
        return None
    nav = positions.get("nav")
    if not isinstance(nav, (int, float)) or nav <= 0:
        return None
    cash = float(positions.get("cash") or 0.0)
    holdings = list(positions.get("holdings", []) or [])

    sleeve_map: dict[str, str] = dict(mandate.get("sleeve_map") or {})
    betas: dict[str, Any] = dict(mandate.get("betas") or {})
    lev: dict[str, Any] = dict(mandate.get("leverage_factors") or {})
    theses: dict[str, Any] = dict(mandate.get("c_theses") or {})
    sleeve_defs: dict[str, Any] = dict(mandate.get("sleeves") or {})

    def _factor(table: dict[str, Any], symbol: str) -> float:
        v = table.get(symbol, table.get("_default", 1.0))
        try:
            return float(v)
        except (TypeError, ValueError):
            return 1.0

    # ── bucket holdings into sleeves ─────────────────────────────────────
    by_sleeve: dict[str, list[dict[str, Any]]] = {}
    for h in holdings:
        symbol = h.get("symbol")
        sleeve_id = sleeve_map.get(symbol, "UNASSIGNED")  # Rule 17.1 fail-closed
        mv = float(h.get("market_value") or 0.0)
        exposure = mv * _factor(betas, symbol) * _factor(lev, symbol)
        row = {
            "symbol": symbol,
            "qty": h.get("qty"),
            "marketValueJpy": round(mv, 2),
            "beta": _factor(betas, symbol),
            "leverageFactor": _factor(lev, symbol),
            "betaAdjExposureJpy": round(exposure, 2),
            "unrealizedReturnPct": h.get("unrealized_return_pct"),
            "flags": [],
        }
        by_sleeve.setdefault(sleeve_id, []).append(row)

    # ── Sleeve C discipline flags (Rule 17.4) ────────────────────────────
    c_def = sleeve_defs.get("C") or {}
    c_rows = by_sleeve.get("C", [])
    c_value = sum(r["marketValueJpy"] for r in c_rows)
    c_cap_frac = float(c_def.get("cap_frac_nav") or 0.20)
    c_cap_breached = c_value > c_cap_frac * nav
    review_frac = float(c_def.get("review_drawdown_frac") or -0.20)
    for r in c_rows:
        entry = theses.get(r["symbol"]) or {}
        if not entry.get("thesis"):
            r["flags"].append(FLAG_THESIS_MISSING)  # fail-closed, Rule 17.4.3
        if c_cap_breached:
            r["flags"].append(FLAG_CAP_BREACHED)
        reunder = entry.get("reunderwrite_price")
        qty = float(r.get("qty") or 0.0)
        price_now = r["marketValueJpy"] / qty if qty > 0 else None
        if reunder and price_now is not None:
            if price_now <= float(reunder) * (1.0 + review_frac):
                r["flags"].append(FLAG_REVIEW_REQUIRED)  # Rule 17.4.4
        # bilateral exit bracket (Rule 17.4.6) — pre-written two-sided close rule.
        upper = entry.get("exit_upper_jpy")
        lower = entry.get("exit_lower_jpy")
        if price_now is not None and (upper is not None or lower is not None):
            if upper is not None and price_now >= float(upper):
                status = "exit_upper"
            elif lower is not None and price_now <= float(lower):
                status = "exit_lower"
            else:
                status = "armed"
            r["exitBracket"] = {
                "upperJpy": float(upper) if upper is not None else None,
                "lowerJpy": float(lower) if lower is not None else None,
                "priceEvaluatedJpy": round(price_now, 2),
                "status": status,
                "note": {
                    "exit_upper": "收盘价触上沿——按预写规则下 S 株市价卖单了结（advice-only，你在券商执行）",
                    "exit_lower": "收盘价触下沿——按预写规则下 S 株市价卖单了结（advice-only，你在券商执行）",
                    "armed": "双边括号已武装——任一收盘触边即终结仓位；触边前不动作",
                }[status],
            }
            if status != "armed":
                r["flags"].append(FLAG_EXIT_TRIGGERED)
        r["reunderwritePrice"] = reunder
        r["thesis"] = entry.get("thesis")
        r["invalidation"] = entry.get("invalidation")

    # ── sleeve summaries ─────────────────────────────────────────────────
    sleeves_out: list[dict[str, Any]] = []
    for sleeve_id in ("A", "B", "C"):
        d = sleeve_defs.get(sleeve_id) or {}
        rows = by_sleeve.get(sleeve_id, [])
        cap_jpy = d.get("cap_jpy")
        current = sum(r["marketValueJpy"] for r in rows)
        entry_out = {
            "id": sleeve_id,
            "role": d.get("role"),
            "label": d.get("label"),
            "expectationLabel": d.get("expectation_label"),  # Rule 17.6
            "targetCapitalJpy": d.get("target_capital_jpy"),
            "capJpy": cap_jpy,
            "capFracNav": d.get("cap_frac_nav"),
            "currentCapitalJpy": round(current, 2),
            "currentExposureJpy": round(sum(r["betaAdjExposureJpy"] for r in rows), 2),
            "holdings": rows,
            "flags": sorted({f for r in rows for f in r["flags"]}),
        }
        target = d.get("target_capital_jpy")
        if isinstance(target, (int, float)):
            entry_out["deploymentGapJpy"] = round(float(target) - current, 2)
        if isinstance(cap_jpy, (int, float)) and current > float(cap_jpy):
            entry_out["flags"] = sorted(set(entry_out["flags"]) | {FLAG_CAP_BREACHED})
        if sleeve_id == "C":
            entry_out["currentFracNav"] = round(c_value / nav, 4)
        if sleeve_id == "B":
            entry_out["precommitment"] = d.get("precommitment")
        sleeves_out.append(entry_out)
    unassigned = by_sleeve.get("UNASSIGNED", [])
    if unassigned:
        sleeves_out.append({
            "id": "UNASSIGNED",
            "label": "未映射（需 sleeve_map 归属）",
            "expectationLabel": None,
            "currentCapitalJpy": round(sum(r["marketValueJpy"] for r in unassigned), 2),
            "currentExposureJpy": round(sum(r["betaAdjExposureJpy"] for r in unassigned), 2),
            "holdings": unassigned,
            "flags": ["unmapped_holdings"],
        })

    # ── totals: exposure band (Rule 17.2) + kill-switch (Rule 17.3) ──────
    all_rows = [r for rows in by_sleeve.values() for r in rows]
    total_exposure = sum(r["betaAdjExposureJpy"] for r in all_rows)
    ratio = total_exposure / nav
    band = mandate.get("exposure_band") or [1.2, 1.6]
    lo, hi = float(band[0]), float(band[1])
    if ratio < lo:
        band_status, band_note = "below_band", (
            f"β调整敞口比 {ratio:.2f}x 低于带 [{lo:.1f}, {hi:.1f}]——A sleeve 尚有未部署额度（advice-only，由你在券商执行）"
        )
    elif ratio > hi:
        band_status, band_note = "above_band", (
            f"β调整敞口比 {ratio:.2f}x 高于带 [{lo:.1f}, {hi:.1f}]——按 Rule 17.2 需减仓回带内；"
            "跌时不减仓会使地板概率推导失效"
        )
    else:
        band_status, band_note = "within_band", f"β调整敞口比 {ratio:.2f}x 在带 [{lo:.1f}, {hi:.1f}] 内"

    floor = float(mandate.get("kill_switch_nav_floor_jpy") or 0.0)
    breached = nav < floor
    kill_switch = {
        "floorJpy": floor,
        "navJpy": round(float(nav), 2),
        "bufferJpy": round(float(nav) - floor, 2),
        "bufferPct": round((float(nav) - floor) / float(nav) * 100, 2),
        "breached": breached,
        "note": (
            "KILL-SWITCH BREACHED — 全部 sleeve 状态=exit + post-mortem（Rule 17.3；系统不能也不会代为平仓）"
            if breached else "距 kill-switch 地板的缓冲"
        ),
    }

    return {
        "disclosure": DISCLOSURE,
        "scoreStatus": "uncalibrated_research_score",
        "mandate": {
            "declaredDate": mandate.get("declared_date"),
            "experimentalCapitalJpy": mandate.get("experimental_capital_jpy"),
            "targetExposureRatio": mandate.get("target_exposure_ratio"),
            "exposureBand": [lo, hi],
            "flagSunsetSessions": int(mandate.get("flag_sunset_sessions") or 7),  # Rule 17.4.7
            "derivationAssumptions": mandate.get("derivation_assumptions"),  # provenance, not prediction
        },
        "navJpy": round(float(nav), 2),
        "cashJpy": round(cash, 2),
        "exposure": {
            "betaAdjustedJpy": round(total_exposure, 2),
            "ratio": round(ratio, 3),
            "bandStatus": band_status,
            "note": band_note,
        },
        "killSwitch": kill_switch,
        "sleeves": sleeves_out,
        "sectorLookThrough": _sector_look_through(all_rows, float(nav), mandate),  # Rule 17.7
    }
