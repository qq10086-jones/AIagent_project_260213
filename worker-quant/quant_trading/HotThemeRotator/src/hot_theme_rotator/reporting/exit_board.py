"""Position Exit Discipline Board (P22-01, Rule 11.17).

Answers 00_DESIGN §2 Q4 — should current holdings take profit, stop out, hold,
or rotate — as pure ARITHMETIC between observed prices (journal-derived
portfolio state, the §14 SSoT) and the discipline that ACTUALLY governs each
holding. It predicts nothing, never acts (Rule 3 — exits happen at the
external broker and are recorded via the manual journal path), and fail-opens
to ``None`` when portfolio data is unavailable — never fabricated holdings
(Rule 11.9.4).

Two discipline layers exist, and precedence matters (Rule 11.17.7, P27):

- The GENERIC swing parameters (avg_cost +2/+3/+5% take-profit, −4% stop) come
  from the "2-5% 止盈换仓" strategy in 00_DESIGN §1. They govern only holdings
  the owner risk mandate does NOT cover.
- The Section 17 / ADR-0012 mandate (``configs/risk_mandate.json``, declared
  2026-07-13) supersedes them for any symbol in ``sleeve_map``. Sleeve A is a
  compensated-beta engine whose expected return IS the equity risk premium
  (Rule 17.1); a cost-anchored stop there realizes the very drawdown the sleeve
  is paid to absorb, and on a 2x instrument a −4% band sits inside two daily
  sigma. Sleeve B is pre-committed to a verdict date (Rule 17.5) — stopping out
  a measurement basket destroys the measurement. Sleeve C is governed by its
  declared bilateral bracket (Rule 17.4.6), which MUST NEVER be the entry cost.

Rendering a cost-anchored −4% against a mandate holding does not merely add
noise: for Sleeve C it re-displays the exact cost anchor Rule 17.4.6 exists to
abolish. So mandate-governed rows suppress the generic refs and show the
governing reference instead.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

DISCLOSURE = (
    "持仓纪律参考 = 你自己声明的止盈/止损参数的算术对照，不是预测、不是卖出指令；"
    "系统无 demonstrated edge；操作在外部券商完成后经手工记录入账 — Rule 3 — "
    "manual execution outside HTR。不含概率/胜率/期望收益。"
)

# Rule 11.17.7 — which layer produced this row's reference levels.
SOURCE_GENERIC = "generic_swing"          # 00_DESIGN §1 params (non-mandate holdings)
SOURCE_SLEEVE_A = "mandate_sleeve_a"      # Rule 17.1/17.2 — band-governed, no per-symbol stop
SOURCE_SLEEVE_B = "mandate_sleeve_b"      # Rule 17.5 — pre-committed to verdict date
SOURCE_BRACKET = "mandate_bracket"        # Rule 17.4.6 — declared bilateral close bracket
SOURCE_REVIEW = "mandate_review_drawdown"  # Rule 17.4.4 — review trigger off re-underwrite
SOURCE_MANDATE_BARE = "mandate_no_level"  # in a sleeve, no declared level to show


@dataclass(frozen=True)
class ExitBoardConfig:
    # Rule 11.17.1 — the generic swing discipline (00_DESIGN §1). Applies only
    # to holdings outside the Section 17 mandate; displayed on the surface,
    # never presented as model targets.
    take_profit_fracs: tuple[float, ...] = (0.02, 0.03, 0.05)
    stop_frac: float = -0.04


def _load_mandate(base_dir: str | Path | None) -> dict[str, Any] | None:
    """Load the owner mandate; ``None`` (fail-open) → generic params govern all."""
    if base_dir is None:
        return None
    try:
        from hot_theme_rotator.risk.sleeve_engine import load_mandate
    except Exception:  # noqa: BLE001 — risk module absent → generic lane, unchanged
        return None
    try:
        return load_mandate(base_dir)
    except Exception:  # noqa: BLE001
        return None


def build_exit_board(
    positions: dict[str, Any] | None,
    *,
    action_board_plan_ready: int | None = None,
    config: ExitBoardConfig = ExitBoardConfig(),
    base_dir: str | Path | None = None,
    mandate: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Assemble the exit board from the serialized portfolio state.

    Returns ``None`` (fail-open, Rule 11.17.5) when the portfolio is
    unavailable or holds nothing — the frontend then renders nothing.

    When the Section 17 mandate is available, holdings mapped to a sleeve are
    governed by it and the generic refs are suppressed for those rows
    (Rule 11.17.7). Without a mandate the board behaves exactly as before.
    """
    if not positions or not positions.get("available"):
        return None
    holdings = list(positions.get("holdings", []) or [])
    if not holdings:
        return None
    if mandate is None:
        mandate = _load_mandate(base_dir)
    sleeve_map: dict[str, str] = dict((mandate or {}).get("sleeve_map") or {})
    rows = []
    for h in holdings:
        try:
            rows.append(_build_row(h, config, mandate, sleeve_map))
        except Exception:  # noqa: BLE001 — one bad holding never breaks the board
            rows.append({
                "symbol": h.get("symbol"), "qty": h.get("qty"),
                "exitStatus": "insufficient_data",
                "statusNote": "行装配失败，按数据不足处理",
                "takeProfitRefs": [], "stopRef": None,
                "sleeve": None, "disciplineSource": SOURCE_GENERIC,
                "mandateRef": None,
            })
    generic_rows = sum(1 for r in rows if r.get("disciplineSource") == SOURCE_GENERIC)
    return {
        "disclosure": DISCLOSURE,
        "scoreStatus": "uncalibrated_research_score",
        "params": {
            "takeProfitFracs": [round(f, 4) for f in config.take_profit_fracs],
            "stopFrac": round(config.stop_frac, 4),
            "basis": "avg_cost",  # references anchor on the recorded cost basis
            # Rule 11.17.7 — say plainly what these params do and do not govern.
            "scope": "non_mandate_holdings",
            "scopeNote": (
                "通用波段参数（00_DESIGN §1）只适用于 Section 17 授权未覆盖的持仓；"
                "sleeve 内持仓由 mandate 自身的纪律管辖"
            ),
            "appliesToRows": generic_rows,
        },
        # Rule 11.17.7 — was a mandate actually loaded for this board?
        "mandateAware": bool(sleeve_map),
        # Rule 11.17.3 — rotate cross-reference is a FACT count, never a directive.
        "actionBoardPlanReady": action_board_plan_ready,
        "rows": rows,
    }


def _base_row(h: dict[str, Any], cost: float, price: float) -> dict[str, Any]:
    return {
        "symbol": h.get("symbol"),
        "qty": h.get("qty"),
        "avgCost": round(cost, 2) if cost else None,
        "marketPrice": round(price, 2) if price else None,
        "marketValue": h.get("market_value"),
        "unrealizedPnl": h.get("unrealized_pnl"),
        "unrealizedReturnPct": h.get("unrealized_return_pct"),
    }


def _build_row(
    h: dict[str, Any],
    config: ExitBoardConfig,
    mandate: dict[str, Any] | None,
    sleeve_map: dict[str, str],
) -> dict[str, Any]:
    symbol = h.get("symbol")
    cost = float(h.get("avg_cost") or 0.0)
    price = float(h.get("market_price") or 0.0)
    sleeve = sleeve_map.get(symbol)
    row = _base_row(h, cost, price)
    row["sleeve"] = sleeve

    # Rule 11.17.2 — fail-closed: no cost basis or price → insufficient_data.
    if cost <= 0 or price <= 0:
        row.update({
            "exitStatus": "insufficient_data",
            "statusNote": "缺少成本或现价，无法做纪律对照",
            "takeProfitRefs": [], "stopRef": None, "mandateRef": None,
            "disciplineSource": SOURCE_GENERIC if sleeve is None else SOURCE_MANDATE_BARE,
        })
        return row

    if sleeve is not None:
        return _mandate_row(row, sleeve, symbol, price, mandate or {})
    return _generic_row(row, config, cost, price)


def _ref(cost: float, price: float, frac: float) -> dict[str, Any]:
    ref_price = round(cost * (1.0 + frac), 2)
    return {
        "frac": round(frac, 4),
        "label": f"止盈参考 {frac:+.0%}" if frac > 0 else f"止损参考 {frac:+.0%}",
        "price": ref_price,
        # distance from CURRENT price to the reference (how far to go)
        "distancePct": round((ref_price - price) / price * 100, 2),
    }


def _generic_row(
    row: dict[str, Any], config: ExitBoardConfig, cost: float, price: float
) -> dict[str, Any]:
    """The 00_DESIGN §1 swing discipline — unchanged, non-mandate holdings only."""
    tp_refs = [_ref(cost, price, f) for f in config.take_profit_fracs]
    stop_ref = _ref(cost, price, config.stop_frac)
    row["takeProfitRefs"] = tp_refs
    row["stopRef"] = stop_ref
    row["mandateRef"] = None
    row["disciplineSource"] = SOURCE_GENERIC

    if price <= stop_ref["price"]:
        status = "stop_reference_breached"
        note = f"现价已低于止损参考 ¥{stop_ref['price']}（成本 {config.stop_frac:+.0%}）"
    elif tp_refs and price >= tp_refs[0]["price"]:
        status = "past_first_take_profit"
        note = f"现价已越过首止盈参考 ¥{tp_refs[0]['price']}（成本 {config.take_profit_fracs[0]:+.0%}）"
    else:
        status = "within_plan"
        note = "现价处于止损参考与首止盈参考之间"
    row["exitStatus"] = status
    row["statusNote"] = note
    return row


def _mandate_row(
    row: dict[str, Any],
    sleeve: str,
    symbol: str,
    price: float,
    mandate: dict[str, Any],
) -> dict[str, Any]:
    """Rule 11.17.7 — Section 17 governs; show the reference that actually binds.

    The generic cost-anchored refs are suppressed here by design: on Sleeve A
    they contradict Rule 17.1 (the drawdown is the compensated risk), and on
    Sleeve C a cost-anchored stop is precisely what Rule 17.4.6 abolishes.
    """
    row["takeProfitRefs"] = []
    row["stopRef"] = None
    row["mandateRef"] = None

    if sleeve == "A":
        row["disciplineSource"] = SOURCE_SLEEVE_A
        row["exitStatus"] = "mandate_governed"
        row["statusNote"] = (
            "Sleeve A 杠杆β引擎：按 Rule 17.1/17.2 无单票止损参考——纪律是组合层"
            "β调整敞口带的再平衡，不是成本锚止损"
        )
        return row

    if sleeve == "B":
        pre = ((mandate.get("sleeves") or {}).get("B") or {}).get("precommitment") or {}
        verdict = pre.get("verdict_date")
        row["disciplineSource"] = SOURCE_SLEEVE_B
        row["exitStatus"] = "mandate_governed"
        row["statusNote"] = (
            "Sleeve B value/E-P 实验：按 Rule 17.5 预承诺至判决日"
            + (f" {verdict}" if verdict else "")
            + "——判决前不因浮亏改尺寸，无单票止损参考"
        )
        return row

    if sleeve == "C":
        return _sleeve_c_row(row, symbol, price, mandate)

    # Mapped to something unrecognized — fail-closed to "no level", never a
    # fabricated generic stop (Rule 17.1 spirit: never silently attributed).
    row["disciplineSource"] = SOURCE_MANDATE_BARE
    row["exitStatus"] = "mandate_governed"
    row["statusNote"] = f"sleeve {sleeve}：mandate 未声明可显示的价位纪律"
    return row


def _sleeve_c_row(
    row: dict[str, Any], symbol: str, price: float, mandate: dict[str, Any]
) -> dict[str, Any]:
    """Sleeve C — the declared bracket (Rule 17.4.6), else the review trigger."""
    entry = (mandate.get("c_theses") or {}).get(symbol) or {}
    upper, lower = entry.get("exit_upper_jpy"), entry.get("exit_lower_jpy")

    def _dist(level: float) -> float:
        return round((float(level) - price) / price * 100, 2)

    if upper is not None or lower is not None:
        breached = (upper is not None and price >= float(upper)) or (
            lower is not None and price <= float(lower)
        )
        row["mandateRef"] = {
            "kind": "bilateral_close_bracket",
            "rule": "17.4.6",
            "label": "双边收盘括号",
            "upperJpy": float(upper) if upper is not None else None,
            "lowerJpy": float(lower) if lower is not None else None,
            "upperDistancePct": _dist(upper) if upper is not None else None,
            "lowerDistancePct": _dist(lower) if lower is not None else None,
            "basis": "declared_close_levels",  # NOT the entry cost — Rule 17.4.6
        }
        row["disciplineSource"] = SOURCE_BRACKET
        if breached:
            row["exitStatus"] = "mandate_exit_triggered"
            row["statusNote"] = (
                "收盘价已触括号边沿——按预写规则了结（advice-only，你在券商执行）"
            )
        else:
            row["exitStatus"] = "mandate_governed"
            row["statusNote"] = (
                "双边括号已武装——任一收盘触边即终结仓位；触边前不动作。"
                "退出价位是预写的收盘价，永不因回到成本而改判（Rule 17.4.6）"
            )
        return row

    reunder = entry.get("reunderwrite_price")
    review_frac = float(
        ((mandate.get("sleeves") or {}).get("C") or {}).get("review_drawdown_frac") or -0.20
    )
    if reunder:
        level = round(float(reunder) * (1.0 + review_frac), 2)
        row["mandateRef"] = {
            "kind": "review_drawdown",
            "rule": "17.4.4",
            "label": f"复核触发参考 {review_frac:+.0%}（自重承保价 ¥{float(reunder):,.0f}）",
            "priceJpy": level,
            "distancePct": _dist(level),
            "basis": "reunderwrite_price",  # NOT the entry cost
        }
        row["disciplineSource"] = SOURCE_REVIEW
        if price <= level:
            row["exitStatus"] = "mandate_review_required"
            row["statusNote"] = (
                f"现价已至复核触发参考 ¥{level}——Rule 17.4.4 要求一个明确的书面决定"
                "（继续持有并写明理由 / 退出）"
            )
        else:
            row["exitStatus"] = "mandate_governed"
            row["statusNote"] = "Sleeve C：未触复核参考；无成本锚止损（Rule 17.4）"
        return row

    row["disciplineSource"] = SOURCE_MANDATE_BARE
    row["exitStatus"] = "mandate_governed"
    row["statusNote"] = (
        "Sleeve C：mandate 未声明退出括号或重承保价——按 Rule 17.4.3 需补写 thesis"
        "（风险授权面板给出 thesis_missing 旗标）"
    )
    return row
