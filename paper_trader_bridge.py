
import json, os, yfinance as yf
from datetime import datetime

def execute_simulated_trade(symbol, quantity, action="BUY"):
    # 1. 加载账户信息
    with open("paper_trading_account.json", "r", encoding="utf-8") as f:
        account = json.load(f)

    # 2. 获取实时价格
    ticker = yf.Ticker(symbol)
    price = ticker.fast_info.last_price or ticker.history(period="1d")['Close'].iloc[-1]
    total_cost = price * quantity
    
    if action == "BUY":
        if account["account_info"]["current_cash"] >= total_cost:
            # 扣款
            account["account_info"]["current_cash"] -= total_cost
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
                "total": round(total_cost, 2)
            })
            print(f"✅ 虚拟执行成功: BUY {quantity} shares of {symbol} at {round(price, 2)}")
        else:
            print("❌ 虚拟资金不足!")
            return

    # 3. 保存更新后的账户
    with open("paper_trading_account.json", "w", encoding="utf-8") as f:
        json.dump(account, f, indent=4, ensure_ascii=False)

if __name__ == "__main__":
    # 根据刚才的分析，我们为 40w 账户买入 1手 (100股) ENEOS
    execute_simulated_trade("5020.T", 100, "BUY")
