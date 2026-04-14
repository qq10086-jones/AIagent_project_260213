from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date, datetime
import sqlite3
from typing import Optional


def _parse_trade_date(value: str) -> Optional[date]:
    text = str(value or "").strip()
    if not text:
        return None
    for fmt in ("%Y-%m-%d", "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S"):
        try:
            return datetime.strptime(text[:19], fmt).date()
        except ValueError:
            continue
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).date()
    except ValueError:
        return None


@dataclass
class KellyPositionSizer:
    win_rate: float
    avg_win: float
    avg_loss: float
    sample_count: int
    kelly_fraction: float = 0.5
    min_position_pct: float = 0.05
    max_position_pct: float = 0.50
    # Default 0.0: when sample_count < min_samples we have no evidence of
    # edge, so the correct Kelly action is no position. A non-zero value
    # must be opted-in explicitly (shadow sizing / bootstrap phase).
    fallback_position_pct: float = 0.0
    min_samples: int = 30
    cooldown_days: int = 5
    cooldown_loss_streak: int = 3
    cooldown_remaining_days: int = 0

    def edge(self) -> float:
        if self.sample_count < self.min_samples:
            return 0.0
        if not (0.0 < self.win_rate < 1.0):
            return 0.0
        if self.avg_win <= 0.0 or self.avg_loss <= 0.0:
            return 0.0
        b = self.avg_win / self.avg_loss
        if b <= 0.0:
            return 0.0
        q = 1.0 - self.win_rate
        value = (self.win_rate * b - q) / b
        return float(value) if value > 0.0 else 0.0

    def suggested_weight(self) -> float:
        if self.cooldown_remaining_days > 0:
            return 0.0
        edge = self.edge()
        if edge <= 0.0:
            if self.sample_count < self.min_samples:
                return min(self.fallback_position_pct, self.max_position_pct)
            return 0.0
        raw = edge * self.kelly_fraction
        return min(max(raw, self.min_position_pct), self.max_position_pct)

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["edge"] = self.edge()
        payload["suggested_weight"] = self.suggested_weight()
        return payload


def compute_kelly_params(
    conn: sqlite3.Connection,
    strategy_id: str,
    lookback_days: int = 60,
    asof: str | None = None,
    fallback_position_pct: float = 0.0,
) -> dict:
    inventory: dict[str, tuple[float, float]] = {}
    realized: list[tuple[date | None, float]] = []
    cutoff = _parse_trade_date(asof) if asof else None
    query = """
        SELECT ts, symbol, side, qty, price
        FROM fills
        WHERE strategy_id=?
    """
    params = [strategy_id]
    if asof:
        query += " AND asof<=?"
        params.append(asof)
    query += " ORDER BY ts, fill_id"
    for ts, symbol, side, qty, price in conn.execute(query, tuple(params)).fetchall():
        trade_date = _parse_trade_date(str(ts))
        if cutoff and trade_date and (cutoff - trade_date).days > lookback_days:
            continue
        side = str(side).upper()
        symbol = str(symbol)
        qty = float(qty or 0.0)
        price = float(price or 0.0)
        cur_qty, cur_cost = inventory.get(symbol, (0.0, 0.0))
        if side == "BUY":
            new_qty = cur_qty + qty
            if new_qty > 0.0:
                cur_cost = (cur_qty * cur_cost + qty * price) / new_qty if cur_qty > 0.0 else price
            inventory[symbol] = (new_qty, cur_cost)
        elif side == "SELL":
            realized_qty = min(cur_qty, qty)
            if realized_qty > 0.0 and cur_cost > 0.0:
                realized.append((trade_date, (price - cur_cost) / cur_cost))
            inventory[symbol] = (max(cur_qty - qty, 0.0), cur_cost)

    returns = [ret for _, ret in realized]
    wins = [ret for ret in returns if ret > 0.0]
    losses = [-ret for ret in returns if ret < 0.0]
    recent_returns = [ret for _, ret in realized[-3:]]

    cooldown_remaining_days = 0
    last_trade_date = realized[-1][0] if realized else None
    if len(recent_returns) == 3 and all(ret < 0.0 for ret in recent_returns):
        if cutoff and last_trade_date:
            elapsed = max((cutoff - last_trade_date).days, 0)
            cooldown_remaining_days = max(5 - elapsed, 0)
        else:
            cooldown_remaining_days = 5

    sizer = KellyPositionSizer(
        win_rate=(len(wins) / len(returns)) if returns else 0.0,
        avg_win=(sum(wins) / len(wins)) if wins else 0.0,
        avg_loss=(sum(losses) / len(losses)) if losses else 0.0,
        sample_count=len(returns),
        cooldown_remaining_days=cooldown_remaining_days,
        fallback_position_pct=float(fallback_position_pct),
    )
    return sizer.to_dict()
