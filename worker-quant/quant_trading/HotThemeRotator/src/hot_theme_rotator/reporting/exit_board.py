"""Position Exit Discipline Board (P22-01, Rule 11.17).

Answers 00_DESIGN §2 Q4 — should current holdings take profit, stop out, hold,
or rotate — as pure ARITHMETIC between observed prices (journal-derived
portfolio state, the §14 SSoT) and the operator's OWN declared discipline
parameters: take-profit references at avg_cost +2/+3/+5% and a stop reference
at avg_cost −4% (the "2-5% 止盈换仓" strategy from 00_DESIGN §1, encoded as
explicit Rule 4 config). It predicts nothing, never acts (Rule 3 — exits
happen at the external broker and are recorded via the manual journal path),
and fail-opens to ``None`` when portfolio data is unavailable — never
fabricated holdings (Rule 11.9.4).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

DISCLOSURE = (
    "持仓纪律参考 = 你自己声明的止盈/止损参数的算术对照，不是预测、不是卖出指令；"
    "系统无 demonstrated edge；操作在外部券商完成后经手工记录入账 — Rule 3 — "
    "manual execution outside HTR。不含概率/胜率/期望收益。"
)


@dataclass(frozen=True)
class ExitBoardConfig:
    # Rule 11.17.1 — the operator's declared discipline (00_DESIGN §1), explicit
    # under Rule 4; displayed on the surface, never presented as model targets.
    take_profit_fracs: tuple[float, ...] = (0.02, 0.03, 0.05)
    stop_frac: float = -0.04


def build_exit_board(
    positions: dict[str, Any] | None,
    *,
    action_board_plan_ready: int | None = None,
    config: ExitBoardConfig = ExitBoardConfig(),
) -> dict[str, Any] | None:
    """Assemble the exit board from the serialized portfolio state.

    Returns ``None`` (fail-open, Rule 11.17.5) when the portfolio is
    unavailable or holds nothing — the frontend then renders nothing.
    """
    if not positions or not positions.get("available"):
        return None
    holdings = list(positions.get("holdings", []) or [])
    if not holdings:
        return None
    rows = []
    for h in holdings:
        try:
            rows.append(_build_row(h, config))
        except Exception:  # noqa: BLE001 — one bad holding never breaks the board
            rows.append({
                "symbol": h.get("symbol"), "qty": h.get("qty"),
                "exitStatus": "insufficient_data",
                "statusNote": "行装配失败，按数据不足处理",
                "takeProfitRefs": [], "stopRef": None,
            })
    return {
        "disclosure": DISCLOSURE,
        "scoreStatus": "uncalibrated_research_score",
        "params": {
            "takeProfitFracs": [round(f, 4) for f in config.take_profit_fracs],
            "stopFrac": round(config.stop_frac, 4),
            "basis": "avg_cost",  # references anchor on the recorded cost basis
        },
        # Rule 11.17.3 — rotate cross-reference is a FACT count, never a directive.
        "actionBoardPlanReady": action_board_plan_ready,
        "rows": rows,
    }


def _build_row(h: dict[str, Any], config: ExitBoardConfig) -> dict[str, Any]:
    symbol = h.get("symbol")
    qty = h.get("qty")
    cost = float(h.get("avg_cost") or 0.0)
    price = float(h.get("market_price") or 0.0)
    row: dict[str, Any] = {
        "symbol": symbol,
        "qty": qty,
        "avgCost": round(cost, 2) if cost else None,
        "marketPrice": round(price, 2) if price else None,
        "marketValue": h.get("market_value"),
        "unrealizedPnl": h.get("unrealized_pnl"),
        "unrealizedReturnPct": h.get("unrealized_return_pct"),
    }
    # Rule 11.17.2 — fail-closed: no cost basis or price → insufficient_data.
    if cost <= 0 or price <= 0:
        row.update({
            "exitStatus": "insufficient_data",
            "statusNote": "缺少成本或现价，无法做纪律对照",
            "takeProfitRefs": [], "stopRef": None,
        })
        return row

    def _ref(frac: float) -> dict[str, Any]:
        ref_price = round(cost * (1.0 + frac), 2)
        return {
            "frac": round(frac, 4),
            "label": f"止盈参考 {frac:+.0%}" if frac > 0 else f"止损参考 {frac:+.0%}",
            "price": ref_price,
            # distance from CURRENT price to the reference (how far to go)
            "distancePct": round((ref_price - price) / price * 100, 2),
        }

    tp_refs = [_ref(f) for f in config.take_profit_fracs]
    stop_ref = _ref(config.stop_frac)
    row["takeProfitRefs"] = tp_refs
    row["stopRef"] = stop_ref

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
