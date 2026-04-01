
import json
from pathlib import Path
from datetime import datetime

import pandas as pd
import yfinance as yf
from scan_market_for_user import scan
from nexus_advanced_strategy import NexusAdvancedStrategyEngine
from paper_trader_bridge import execute_simulated_trade, load_paper_account, load_paper_config


PROJECT_ROOT = Path(__file__).resolve().parent
QUANT_REPORTS_DIR = PROJECT_ROOT / "worker-quant" / "quant_trading" / "Project_optimized" / "reports"
QUANT_ARTIFACTS_DIR = PROJECT_ROOT / "worker-quant" / "quant_trading" / "Project_optimized" / "artifacts" / "decision"


def _load_formal_target_weights(max_age_days=None):
    target_path = QUANT_REPORTS_DIR / "target_weights.csv"
    meta_path = QUANT_REPORTS_DIR / "target_weights_meta.json"
    if not target_path.exists():
        return None
    try:
        df = pd.read_csv(target_path)
    except Exception:
        return None
    if "symbol" not in df.columns or "target_weight" not in df.columns:
        return None
    df = df[["symbol", "target_weight"]].copy()
    df["symbol"] = df["symbol"].astype(str)
    df["target_weight"] = pd.to_numeric(df["target_weight"], errors="coerce").fillna(0.0)
    if meta_path.exists() and max_age_days is not None:
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            exported_asof = meta.get("exported_asof")
            if exported_asof:
                age_days = (datetime.now().date() - datetime.strptime(str(exported_asof), "%Y-%m-%d").date()).days
                if age_days > int(max_age_days):
                    print(f"⏳ 正式量化目标过期: exported_asof={exported_asof} age={age_days}d > {max_age_days}d")
                    return None
        except Exception:
            pass
    return df


def _latest_formal_orders():
    if not QUANT_ARTIFACTS_DIR.exists():
        return None
    order_files = sorted(
        QUANT_ARTIFACTS_DIR.glob("*/*/orders_proposal.csv"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not order_files:
        return None
    latest = order_files[0]
    try:
        df = pd.read_csv(latest)
    except Exception:
        return None
    if df.empty or "symbol" not in df.columns or "side" not in df.columns or "qty" not in df.columns:
        return None
    return latest, df


def _execute_formal_order_file():
    payload = _latest_formal_orders()
    if payload is None:
        return False
    path, df = payload
    df = df.copy()
    df["qty"] = pd.to_numeric(df["qty"], errors="coerce").fillna(0).astype(int)
    df["side"] = df["side"].astype(str).str.upper()
    df = df[(df["qty"] > 0) & (df["side"].isin(["BUY", "SELL"]))]
    if df.empty:
        print(f"📭 最新正式订单文件为空: {path}")
        return False

    print(f"🧠 优先执行正式订单文件: {path}")
    for _, row in df.iterrows():
        print(f"📌 正式订单: {row['side']} {int(row['qty'])} {row['symbol']}")
        execute_simulated_trade(str(row["symbol"]), int(row["qty"]), str(row["side"]))
    return True


def _formal_order_plan(account, target_df, min_cash_buffer_pct):
    positions = account.get("positions", {})
    cash = float(account["account_info"]["current_cash"])
    prices = {}
    nav = cash
    universe = set(target_df["symbol"].tolist()) | set(positions.keys())

    for symbol in sorted(universe):
        try:
            price = float(yf.Ticker(symbol).fast_info.last_price)
        except Exception:
            continue
        prices[symbol] = price
        if symbol in positions:
            nav += float(positions[symbol]["qty"]) * price

    if not prices:
        return []

    weight_map = {str(r["symbol"]): max(float(r["target_weight"]), 0.0) for _, r in target_df.iterrows()}
    wsum = float(sum(weight_map.values()))
    if wsum <= 1e-9:
        return []
    weight_map = {sym: w / wsum for sym, w in weight_map.items()}

    target_investable_nav = nav * (1.0 - float(min_cash_buffer_pct))
    plans = []
    for symbol in sorted(universe):
        if symbol not in prices:
            continue
        cur_qty = int(positions.get(symbol, {}).get("qty", 0))
        target_weight = float(weight_map.get(symbol, 0.0))
        target_value = target_investable_nav * target_weight
        target_qty = int(target_value // (prices[symbol] * 100)) * 100
        diff = target_qty - cur_qty
        if diff == 0:
            continue
        plans.append({
            "symbol": symbol,
            "side": "BUY" if diff > 0 else "SELL",
            "qty": abs(int(diff)),
            "price": prices[symbol],
            "target_weight": target_weight,
        })

    plans.sort(key=lambda x: (0 if x["side"] == "SELL" else 1, -x["qty"] * x["price"]))
    return plans


def _execute_formal_quant_plan(account, min_cash_buffer_pct, max_age_days):
    if _execute_formal_order_file():
        return True

    target_df = _load_formal_target_weights(max_age_days=max_age_days)
    if target_df is None:
        return False

    plans = _formal_order_plan(account, target_df, min_cash_buffer_pct=min_cash_buffer_pct)
    if not plans:
        print("📭 正式量化目标已存在，但当前账户与目标仓位一致或目标不可执行。")
        return True

    print(f"🧠 检测到正式量化目标仓位，优先执行目标再平衡。候选订单数: {len(plans)}")
    for plan in plans:
        print(
            f"📌 正式计划: {plan['side']} {plan['qty']} {plan['symbol']} "
            f"| px≈{plan['price']:.2f} | target_w={plan['target_weight']:.2%}"
        )
        execute_simulated_trade(plan["symbol"], int(plan["qty"]), plan["side"])
    return True


def autonomous_decision():
    # 1. 获取账户现状
    account = load_paper_account()
    config = load_paper_config()
    strategy_cfg = config.get("strategy", {})

    cash = float(account["account_info"]["current_cash"])
    initial_capital = float(account["account_info"]["initial_capital"])
    held_symbols = set(account["positions"].keys())
    min_cash_buffer_pct = float(strategy_cfg.get("min_cash_buffer_pct", 0.2))
    max_formal_signal_age_days = int(strategy_cfg.get("max_formal_signal_age_days", 5))
    max_new_positions = int(strategy_cfg.get("max_new_positions_per_run", 1))
    max_total_positions = int(strategy_cfg.get("max_total_positions", 4))
    stop_loss_pct = float(strategy_cfg.get("stop_loss_pct", 0.05))
    take_profit_pct = float(strategy_cfg.get("take_profit_pct", 0.08))
    trim_profit_pct = float(strategy_cfg.get("rebalance_trim_profit_pct", 0.03))
    min_candidate_score = float(strategy_cfg.get("min_candidate_score", 45.0))
    oversold_rsi = float(strategy_cfg.get("prefer_oversold_rsi_below", 35.0))
    risk_tolerance = str(strategy_cfg.get("risk_tolerance", "MEDIUM")).upper()
    engine = NexusAdvancedStrategyEngine(capital=initial_capital, risk_tolerance=risk_tolerance)
    
    print(f"🤖 Worker-Quant 启动自主决策程序...")
    print(f"💰 当前可用虚拟资金: {cash} JPY")
    print(f"🧭 风险偏好: {risk_tolerance} | 最低候选分: {min_candidate_score}")

    if _execute_formal_quant_plan(
        account,
        min_cash_buffer_pct=min_cash_buffer_pct,
        max_age_days=max_formal_signal_age_days,
    ):
        return
    
    # 2. 扫描市场获取候选名单
    findings = scan(remaining_cash=cash)
    findings_by_symbol = {f["symbol"]: f for f in findings}

    # 3. 先做卖出纪律
    sells = 0
    for symbol, pos in list(account.get("positions", {}).items()):
        current_price = None
        if symbol in findings_by_symbol:
            current_price = float(findings_by_symbol[symbol]["price"])
        else:
            try:
                current_price = float(yf.Ticker(symbol).fast_info.last_price)
            except Exception:
                current_price = None
        if current_price is None:
            continue
        avg_price = float(pos.get("avg_price", 0.0) or 0.0)
        if avg_price <= 0:
            continue
        pnl_pct = current_price / avg_price - 1.0
        if pnl_pct <= -stop_loss_pct or pnl_pct >= take_profit_pct:
            reason = "止损" if pnl_pct <= -stop_loss_pct else "止盈"
            print(f"🛑 {reason}触发: {symbol} | PnL={pnl_pct:+.2%} | 卖出 {int(pos['qty'])} 股")
            execute_simulated_trade(symbol, int(pos["qty"]), "SELL")
            sells += 1
        elif pnl_pct >= trim_profit_pct and cash < initial_capital * min_cash_buffer_pct:
            trim_qty = max(100, int(pos["qty"]) // 2 // 100 * 100)
            trim_qty = min(trim_qty, int(pos["qty"]))
            print(f"✂️ 浮盈再平衡: {symbol} | PnL={pnl_pct:+.2%} | 卖出 {trim_qty} 股补现金")
            execute_simulated_trade(symbol, trim_qty, "SELL")
            sells += 1

    if sells:
        account = load_paper_account()
        cash = float(account["account_info"]["current_cash"])
        held_symbols = set(account["positions"].keys())

    # 4. 买入候选筛选
    regime = engine.market_regime_filter()
    print(f"🌐 市场状态: {regime}")
    if regime == "BEAR":
        print("⏸️ 当前处于 BEAR 状态，停止新增仓位，仅执行卖出纪律。")
        return

    candidates = []
    reserve_cash = initial_capital * min_cash_buffer_pct
    available_for_new = max(cash - reserve_cash, 0.0)
    open_slots = max(max_total_positions - len(held_symbols), 0)
    if open_slots <= 0:
        print("⏸️ 当前持仓数已达上限，保持仓位。")
        return

    for f in findings:
        if f["symbol"] in held_symbols:
            continue
        if f["lot_cost"] > available_for_new:
            continue
        review = engine.evaluate_candidate(f["symbol"], current_cash=cash, news_context=f"{f['name']} oversold scan")
        if review.get("signal") != "BUY":
            continue
        score = max(0.0, (oversold_rsi - float(f["rsi"])) * 2.0) + max(0.0, -float(f["bias"])) * 1.5 + float(review["metrics"]["historical_win_rate"]) * 20.0
        item = {
            "symbol": f["symbol"],
            "name": f["name"],
            "score": round(score, 2),
            "scan": f,
            "review": review,
        }
        if item["score"] >= min_candidate_score:
            candidates.append(item)

    candidates.sort(key=lambda x: x["score"], reverse=True)
    if not candidates:
        print("⏸️ 当前没有同时满足风险门控与评分阈值的建仓机会，保持现金观望。")
        return

    buy_count = 0
    for candidate in candidates[: min(max_new_positions, open_slots)]:
        action = candidate["review"]["action"]
        qty = max(100, int(action["shares"]) // 100 * 100)
        if qty <= 0:
            continue
        if candidate["scan"]["lot_cost"] > cash * (1.0 - min_cash_buffer_pct):
            continue
        print(
            f"🎯 决策选定: {candidate['name']} ({candidate['symbol']}) | "
            f"score={candidate['score']} | stop={action['stop_loss']} | tp={action['take_profit']}"
        )
        print(f"📊 理由: {candidate['review']['reason']}")
        execute_simulated_trade(candidate["symbol"], qty, "BUY")
        buy_count += 1
        account = load_paper_account()
        cash = float(account["account_info"]["current_cash"])

    if buy_count == 0:
        print("⏸️ 候选存在，但资金缓冲约束阻止了新增建仓。")

if __name__ == "__main__":
    autonomous_decision()
