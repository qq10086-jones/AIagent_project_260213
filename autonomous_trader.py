
import json, os, yfinance as yf
from datetime import datetime
from scan_market_for_user import scan
from paper_trader_bridge import execute_simulated_trade

def autonomous_decision():
    # 1. 获取账户现状
    with open("paper_trading_account.json", "r", encoding="utf-8") as f:
        account = json.load(f)
    
    cash = account["account_info"]["current_cash"]
    held_symbols = account["positions"].keys()
    
    print(f"🤖 Worker-Quant 启动自主决策程序...")
    print(f"💰 当前可用虚拟资金: {cash} JPY")
    
    # 2. 扫描市场获取候选名单
    findings = scan()
    
    # 3. 决策逻辑：
    # - 必须是扫描结果中 RSI 最低的（最超卖）
    # - 必须不在当前持仓中（分散风险）
    # - 成本必须小于现金的 80% (预留部分现金)
    
    decision = None
    for f in findings:
        if f["symbol"] in held_symbols:
            continue
        if f["lot_cost"] <= cash * 0.8:
            decision = f
            break
            
    if decision:
        print(f"🎯 决策选定: {decision['name']} ({decision['symbol']})")
        print(f"📊 理由: RSI({decision['rsi']}) 极低，属于强力超跌反弹标的。")
        
        # 执行买入 1 手 (100股)
        execute_simulated_trade(decision["symbol"], 100, "BUY")
        
        # 如果资金还充裕，看看能不能再买一个极其便宜的（分散风险）
        # 刷新一下账户现金
        with open("paper_trading_account.json", "r", encoding="utf-8") as f:
            new_account = json.load(f)
        new_cash = new_account["account_info"]["current_cash"]
        
        for f in findings:
            if f["symbol"] in new_account["positions"] or f["symbol"] == decision["symbol"]:
                continue
            if f["lot_cost"] < 50000 and f["lot_cost"] <= new_cash * 0.5:
                print(f"🎯 额外决策: 买入廉价标的 {f['name']} ({f['symbol']}) 分散风险。")
                execute_simulated_trade(f["symbol"], 100, "BUY")
                break
    else:
        print("⏸️ 当前没有符合量化条件的建仓机会，保持现金观望。")

if __name__ == "__main__":
    autonomous_decision()
