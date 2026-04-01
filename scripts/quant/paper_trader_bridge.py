
import json, os, sys, yfinance as yf
from hashlib import sha1
from datetime import datetime, time
from pathlib import Path
from zoneinfo import ZoneInfo

PROJECT_ROOT = Path(__file__).resolve().parent
PROJECT_OPT_DIR = PROJECT_ROOT / "worker-quant" / "quant_trading" / "Project_optimized"
if str(PROJECT_OPT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_OPT_DIR))

from trade_schema import connect, ensure_trade_tables

DEFAULT_PAPER_DB = PROJECT_OPT_DIR / "japan_market.db"
DEFAULT_PAPER_CONFIG = PROJECT_ROOT / "paper_trading_config.json"
JST = ZoneInfo("Asia/Tokyo")


def get_paper_db_path():
    return Path(os.getenv("PAPER_TRADE_DB", str(DEFAULT_PAPER_DB)))


def get_paper_config_path():
    return Path(os.getenv("PAPER_TRADE_CONFIG", str(DEFAULT_PAPER_CONFIG)))


def load_paper_account():
    with open("paper_trading_account.json", "r", encoding="utf-8") as f:
        account = json.load(f)
    account.setdefault("positions", {})
    account.setdefault("trade_history", [])
    account.setdefault("pending_orders", [])
    return account


def load_paper_config():
    path = get_paper_config_path()
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {
        "goals": {
            "weekly_floor_return_pct": 1.0,
            "weekly_target_return_pct": 2.0,
            "annual_target_return_low_pct": 300.0,
            "annual_target_return_high_pct": 400.0,
            "rolling_windows_weeks": [2, 4, 8],
        },
        "strategy": {
            "risk_tolerance": "MEDIUM",
            "max_formal_signal_age_days": 5,
            "min_cash_buffer_pct": 0.2,
            "max_new_positions_per_run": 1,
            "max_total_positions": 4,
            "take_profit_pct": 0.08,
            "stop_loss_pct": 0.05,
            "rebalance_trim_profit_pct": 0.03,
            "min_candidate_score": 45.0,
            "prefer_oversold_rsi_below": 35.0,
        },
    }


def save_paper_account(account):
    with open("paper_trading_account.json", "w", encoding="utf-8") as f:
        json.dump(account, f, indent=4, ensure_ascii=False)


def _paper_fee_bps():
    cfg = load_paper_config()
    exec_cfg = cfg.get("execution", {})
    if "fee_bps" in exec_cfg:
        return float(exec_cfg.get("fee_bps", 10.0))
    strategy_cfg = cfg.get("strategy", {})
    return float(strategy_cfg.get("fee_bps", 10.0))


def _trade_fee(total_notional):
    return round(float(total_notional) * (_paper_fee_bps() / 10000.0), 2)


def _paper_run_id(trade):
    raw = f"{trade['date']}|{trade['action']}|{trade['symbol']}|{trade['qty']}|{trade['price']}"
    return "paper_" + sha1(raw.encode("utf-8")).hexdigest()[:12]


def _fetch_market_price(symbol, fallback_price=None):
    try:
        ticker = yf.Ticker(symbol)
        price = ticker.fast_info.last_price or ticker.history(period="1d")["Close"].iloc[-1]
        return float(price)
    except Exception:
        return float(fallback_price) if fallback_price is not None else None


def _fetch_db_close(conn, symbol, asof):
    try:
        row = conn.execute(
            "SELECT close FROM daily_prices WHERE symbol=? AND date=?",
            (str(symbol), str(asof)),
        ).fetchone()
    except Exception:
        return None
    if not row or row[0] is None:
        return None
    try:
        return float(row[0])
    except Exception:
        return None


def _now_jst():
    return datetime.now(JST)


def _is_trading_session_open(now_jst=None):
    now_jst = now_jst or _now_jst()
    if now_jst.weekday() >= 5:
        return False
    t = now_jst.time()
    morning_open = time(9, 0)
    morning_close = time(11, 30)
    afternoon_open = time(12, 30)
    afternoon_close = time(15, 30)
    return (morning_open <= t <= morning_close) or (afternoon_open <= t <= afternoon_close)


def sync_account_to_db(account=None):
    account = account or load_paper_account()
    db_path = get_paper_db_path()
    conn = connect(str(db_path))
    ensure_trade_tables(conn)

    try:
        trade_history = list(account.get("trade_history") or [])
        latest_trade = trade_history[-1] if trade_history else None
        latest_asof = (
            str(latest_trade["date"])[:10]
            if latest_trade and latest_trade.get("date")
            else str(account.get("account_info", {}).get("start_date") or datetime.now().strftime("%Y-%m-%d"))
        )

        with conn:
            conn.execute("DELETE FROM fills WHERE source='paper_trader_bridge'")
            conn.execute("DELETE FROM decision_runs WHERE run_id LIKE 'paper_%'")
            conn.execute("DELETE FROM account_snapshots WHERE run_id LIKE 'paper_%'")
            for trade in trade_history:
                run_id = _paper_run_id(trade)
                asof = str(trade["date"])[:10]
                fee = float(trade.get("fee", 0.0) or 0.0)
                conn.execute(
                    """
                    INSERT OR REPLACE INTO decision_runs(run_id, asof, ts, snapshot_path, status, notes)
                    VALUES (?, ?, ?, NULL, 'filled', ?)
                    """,
                    (
                        run_id,
                        asof,
                        str(trade["date"]),
                        "paper simulated trade",
                    ),
                )
                fill_id = sha1(
                    f"{run_id}|{trade['date']}|{trade['symbol']}|{trade['action']}|{trade['qty']}|{trade['price']}".encode("utf-8")
                ).hexdigest()[:16]
                conn.execute(
                    """
                    INSERT OR REPLACE INTO fills(
                      fill_id, order_id, run_id, asof, ts, symbol, side, qty, price, fee, tax, venue, external_ref, source
                    ) VALUES (?, NULL, ?, ?, ?, ?, ?, ?, ?, ?, 0, 'PAPER', '', 'paper_trader_bridge')
                    """,
                    (
                        fill_id,
                        run_id,
                        asof,
                        str(trade["date"]),
                        str(trade["symbol"]),
                        str(trade["action"]).upper(),
                        float(trade["qty"]),
                        float(trade["price"]),
                        fee,
                    ),
                )

            conn.execute("DELETE FROM positions WHERE asof=?", (latest_asof,))

            positions_value = 0.0
            for symbol, pos in sorted((account.get("positions") or {}).items()):
                avg_price = float(pos.get("avg_price", 0.0) or 0.0)
                market_price = _fetch_db_close(conn, symbol, latest_asof)
                if market_price is None:
                    market_price = _fetch_market_price(symbol, fallback_price=avg_price)
                if market_price is None:
                    market_value = None
                    unrealized_pnl = None
                else:
                    market_value = float(pos["qty"]) * market_price
                    unrealized_pnl = (market_price - avg_price) * float(pos["qty"])
                    positions_value += market_value

                conn.execute(
                    """
                    INSERT INTO positions(asof, symbol, qty, avg_cost, market_price, market_value, unrealized_pnl)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        latest_asof,
                        symbol,
                        float(pos["qty"]),
                        avg_price,
                        market_price,
                        market_value,
                        unrealized_pnl,
                    ),
                )

            cash = float(account.get("account_info", {}).get("current_cash", 0.0) or 0.0)
            nav = cash + positions_value
            latest_run_id = _paper_run_id(latest_trade) if latest_trade else "paper_state_sync"
            total_fees = sum(float(t.get("fee", 0.0) or 0.0) for t in trade_history)
            notes = (
                f"paper sync; initial_capital={account.get('account_info', {}).get('initial_capital', 0)}; "
                f"trade_count={len(trade_history)}; pending_count={len(account.get('pending_orders', []))}"
            )
            conn.execute(
                """
                INSERT OR REPLACE INTO account_snapshots(
                  asof, ts, run_id, cash, positions_value, nav, net_trade_cashflow, fees, tax, notes
                ) VALUES (?, ?, ?, ?, ?, ?, NULL, ?, 0, ?)
                """,
                (
                    latest_asof,
                    datetime.now().isoformat(timespec="seconds"),
                    latest_run_id,
                    cash,
                    positions_value,
                    nav,
                    total_fees,
                    notes,
                ),
            )

        return {
            "db_path": str(db_path),
            "asof": latest_asof,
            "cash": cash,
            "positions_value": positions_value,
            "nav": nav,
            "trade_count": len(trade_history),
        }
    finally:
        conn.close()


def _queue_pending_order(account, symbol, quantity, action):
    pending_orders = account.setdefault("pending_orders", [])
    for existing in pending_orders:
        if (
            str(existing.get("symbol")) == str(symbol)
            and int(existing.get("qty", 0)) == int(quantity)
            and str(existing.get("action", "")).upper() == str(action).upper()
            and str(existing.get("status", "")).upper() == "PENDING_MARKET_OPEN"
        ):
            print(f"🕒 已存在待执行订单，跳过重复挂单: {action} {quantity} {symbol}")
            return
    order = {
        "created_at": _now_jst().strftime("%Y-%m-%d %H:%M:%S"),
        "symbol": symbol,
        "qty": int(quantity),
        "action": str(action).upper(),
        "status": "PENDING_MARKET_OPEN",
    }
    pending_orders.append(order)
    save_paper_account(account)
    sync_account_to_db(account)
    print(f"🕒 非交易时段，已挂起订单: {order['action']} {order['qty']} {symbol}")

def execute_simulated_trade(symbol, quantity, action="BUY"):
    # 1. 加载账户信息
    account = load_paper_account()
    action = str(action).upper()

    if not _is_trading_session_open():
        _queue_pending_order(account, symbol, quantity, action)
        return

    # 2. 获取实时价格
    fallback_price = None
    pos = account.get("positions", {}).get(symbol)
    if pos:
        fallback_price = pos.get("avg_price")
    price = _fetch_market_price(symbol, fallback_price=fallback_price)
    if price is None:
        print("❌ 无法获取市场价格!")
        return
    total_cost = price * quantity
    fee = _trade_fee(total_cost)

    if action == "BUY":
        gross_cash_out = total_cost + fee
        if account["account_info"]["current_cash"] >= gross_cash_out:
            # 扣款
            account["account_info"]["current_cash"] -= gross_cash_out
            # 加仓
            pos = account["positions"].get(symbol, {"qty": 0, "avg_price": 0})
            new_qty = pos["qty"] + quantity
            new_avg = (pos["avg_price"] * pos["qty"] + total_cost) / new_qty
            account["positions"][symbol] = {"qty": new_qty, "avg_price": round(new_avg, 2)}
            # 记录历史
            account["trade_history"].append({
                "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "action": "BUY",
                "symbol": symbol,
                "qty": quantity,
                "price": round(price, 2),
                "total": round(total_cost, 2),
                "fee": fee,
            })
            print(f"✅ 虚拟执行成功: BUY {quantity} shares of {symbol} at {round(price, 2)} fee={fee:.2f}")
        else:
            print("❌ 虚拟资金不足!")
            return
    elif action == "SELL":
        pos = account["positions"].get(symbol)
        if not pos or pos["qty"] < quantity:
            print("❌ 可卖持仓不足!")
            return
        net_cash_in = total_cost - fee
        account["account_info"]["current_cash"] += net_cash_in
        remaining_qty = pos["qty"] - quantity
        if remaining_qty > 0:
            account["positions"][symbol] = {"qty": remaining_qty, "avg_price": pos["avg_price"]}
        else:
            account["positions"].pop(symbol, None)
        account["trade_history"].append({
            "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "action": "SELL",
            "symbol": symbol,
            "qty": quantity,
            "price": round(price, 2),
            "total": round(total_cost, 2),
            "fee": fee,
        })
        print(f"✅ 虚拟执行成功: SELL {quantity} shares of {symbol} at {round(price, 2)} fee={fee:.2f}")
    else:
        print(f"❌ 不支持的交易动作: {action}")
        return

    # 3. 保存更新后的账户
    save_paper_account(account)
    sync_result = sync_account_to_db(account)
    print(
        f"🧾 已同步正式账本: asof={sync_result['asof']} nav={sync_result['nav']:.0f} "
        f"db={sync_result['db_path']}"
    )

if __name__ == "__main__":
    # 根据刚才的分析，我们为 40w 账户买入 1手 (100股) ENEOS
    execute_simulated_trade("5020.T", 100, "BUY")
