
import os, sys, json, yfinance as yf
import pandas as pd
import numpy as np

# 选取的扫描池 (从 db_update.py 中提取的具有代表性的标的)
SCAN_POOL = [
    ("4063.T", "Shin-Etsu"), ("6723.T", "Renesas"), ("6758.T", "Sony"), 
    ("6501.T", "Hitachi"), ("6752.T", "Panasonic"), ("7751.T", "Canon"),
    ("7011.T", "Mitsubishi Heavy"), ("8306.T", "MUFG"), ("8316.T", "SMBC"),
    ("8411.T", "Mizuho"), ("8766.T", "Tokio Marine"), ("8591.T", "ORIX"),
    ("7203.T", "Toyota"), ("7267.T", "Honda"), ("7201.T", "Nissan"),
    ("9101.T", "NYK Line"), ("1605.T", "Inpex"), ("9432.T", "NTT"),
    ("9984.T", "SoftBank Group"), ("6702.T", "Fujitsu"), ("6301.T", "Komatsu"),
    ("8058.T", "Mitsubishi Corp"), ("8001.T", "Itochu"), ("4502.T", "Takeda"),
    ("2914.T", "Japan Tobacco"), ("6902.T", "Denso"), ("6503.T", "Mitsubishi Electric"),
    ("4901.T", "Fujifilm"), ("7269.T", "Suzuki"), ("9501.T", "TEPCO")
]

def scan(remaining_cash=None):
    if remaining_cash is None:
        remaining_cash = 271300.0
    results = []
    print(f"🔍 正在扫描市场，剩余预算: {remaining_cash} JPY...")
    
    for symbol, name in SCAN_POOL:
        try:
            t = yf.Ticker(symbol)
            hist = t.history(period="3mo")
            if hist.empty: continue
            
            price = hist['Close'].iloc[-1]
            lot_cost = price * 100
            
            # 过滤资金不足的股票
            if lot_cost > remaining_cash:
                continue
                
            # 计算 RSI
            delta = hist['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rsi = 100 - (100 / (1 + (gain.iloc[-1] / loss.iloc[-1])))
            
            # 计算 Bias (MA20)
            ma20 = hist['Close'].rolling(window=20).mean().iloc[-1]
            bias = (price - ma20) / ma20
            
            results.append({
                "symbol": symbol,
                "name": name,
                "price": round(price, 2),
                "lot_cost": round(lot_cost, 0),
                "rsi": round(rsi, 2),
                "bias": round(bias * 100, 2),
                "change": round((hist['Close'].iloc[-1]/hist['Close'].iloc[-2]-1)*100, 2)
            })
        except:
            continue
            
    # 排序逻辑：优先寻找 RSI 低 (超卖) 且 Bias 低 (远离均线) 的
    # 或者寻找股息/稳健标的
    results.sort(key=lambda x: x['rsi'])
    return results

if __name__ == "__main__":
    findings = scan()
    # 打印前 5 个最值得关注的
    print("\n--- 潜力个股推荐 (基于 27w 剩余预算) ---")
    for f in findings[:8]:
        status = "⚠️ 极度超卖" if f['rsi'] < 35 else "📉 回调中"
        print(f"[{f['symbol']}] {f['name']:<15} | 价格: {f['price']:<7} | 1手成本: {f['lot_cost']:<8} | RSI: {f['rsi']:<5} | Bias: {f['bias']:>5}% | {status}")
