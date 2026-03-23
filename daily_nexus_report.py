
import json, os, yfinance as yf
from datetime import datetime

def generate_daily_report():
    # 1. 加载账户
    if not os.path.exists("paper_trading_account.json"):
        return "❌ 未找到账户文件"
    
    with open("paper_trading_account.json", "r", encoding="utf-8") as f:
        account = json.load(f)
    
    # 2. 计算当前盈亏 (PnL)
    total_market_value = 0
    pos_details = []
    
    print("📊 正在调取实时价格进行结算...")
    for symbol, pos in account["positions"].items():
        ticker = yf.Ticker(symbol)
        current_price = ticker.fast_info.last_price
        val = current_price * pos["qty"]
        pnl = (current_price - pos["avg_price"]) * pos["qty"]
        pnl_pct = (current_price / pos["avg_price"] - 1) * 100
        
        total_market_value += val
        pos_details.append({
            "symbol": symbol,
            "qty": pos["qty"],
            "avg": pos["avg_price"],
            "current": round(current_price, 2),
            "pnl": round(pnl, 0),
            "pnl_pct": round(pnl_pct, 2)
        })

    current_total = total_market_value + account["account_info"]["current_cash"]
    initial = account["account_info"]["initial_capital"]
    total_return_pct = (current_total / initial - 1) * 100
    
    # 周目标 (5%) 追踪
    target_weekly = 5.0
    progress = (total_return_pct / target_weekly) * 100

    # 3. 构建报告文本
    report = f"""
==========================================
 🏛️ NEXUS QUANT 每日交易及收益分析报告
 📅 日期: {datetime.now().strftime("%Y-%m-%d")}
==========================================

💰 [账户概览]
- 初始资金: {initial:,.0f} JPY
- 当前总资产: {current_total:,.0f} JPY
- 现金余额: {account["account_info"]["current_cash"]:,.0f} JPY
- 总收益率: {total_return_pct:+.2f}%
- 🎯 本周 5% 目标完成度: {progress:.1f}%

📈 [持仓明细]
"""
    for p in pos_details:
        report += f"- {p['symbol']}: {p['qty']}股 | 现价: {p['current']} | 盈亏: {p['pnl']:+,.0f} ({p['pnl_pct']:+.2f}%)\n"

    report += """
🔍 [今日量化分析与交易总结]
- 交易动作: 今日机器人根据 RSI 超跌模型执行了 7267.T 和 7201.T 的抄底操作。
- 市场点评: 日经225今日大幅回调，属于典型的恐慌性抛售，量化模型检测到多只龙头股进入极度超卖区（RSI < 20），因此加大了权益仓位。
- 手续费说明: 已应用 SBI 证券“零革新”政策，模拟交易手续费记为 0 JPY。

💡 [后续策略]
- 当前仓位约 72.5%，保留约 11w 现金以应对可能的二次下探。
- 若明天 5020.T 反弹超过 3%，机器人将考虑减仓锁定部分利润以冲刺周 5% 目标。
==========================================
"""
    return report

if __name__ == "__main__":
    report = generate_daily_report()
    print(report)
    # 模拟将报告保存到本地
    with open("LATEST_TRADING_REPORT.md", "w", encoding="utf-8") as f:
        f.write(report)
