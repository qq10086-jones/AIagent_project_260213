
import os, sys, json
import yfinance as yf
import pandas as pd
import numpy as np

def analyze_symbol(symbol="5020.T"):
    ticker = yf.Ticker(symbol)
    
    # 1. 基础信息
    info = ticker.info
    hist = ticker.history(period="6mo")
    
    if hist.empty:
        return {"error": "No data found"}

    current_price = hist['Close'].iloc[-1]
    prev_close = hist['Close'].iloc[-2]
    change_pct = (current_price / prev_close - 1) * 100
    
    # 2. 技术指标简析
    # 20日均线
    ma20 = hist['Close'].rolling(window=20).mean().iloc[-1]
    # RSI (14)
    delta = hist['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs.iloc[-1]))
    
    # 3. 财务概览 (ENEOS 特色)
    div_yield = info.get('dividendYield', 0) * 100
    pe_ratio = info.get('trailingPE', 0)
    pb_ratio = info.get('priceToBook', 0)
    
    result = {
        "symbol": symbol,
        "name": info.get('longName', 'ENEOS Holdings, Inc.'),
        "price": round(current_price, 2),
        "change_pct": round(change_pct, 2),
        "ma20_dist": round((current_price / ma20 - 1) * 100, 2) if ma20 else 0,
        "rsi": round(rsi, 2),
        "div_yield": round(div_yield, 2),
        "pe": round(pe_ratio, 2),
        "pb": round(pb_ratio, 2),
        "market_cap": info.get('marketCap', 0)
    }
    return result

if __name__ == "__main__":
    data = analyze_symbol("5020.T")
    print(json.dumps(data, indent=2))
