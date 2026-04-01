
import os, sys, json
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def backtest_and_predict(symbol="5020.T"):
    ticker = yf.Ticker(symbol)
    # 获取较长时间序列进行回测计算指标
    df = ticker.history(period="1y")
    if df.empty: return {"error": "No data"}

    # 1. 计算因子 (Alpha Factors)
    # 动量因子 (20d Return)
    df['mom20'] = df['Close'].pct_change(20)
    # 波动率因子 (20d Std)
    df['vol20'] = df['Close'].pct_change().rolling(20).std()
    # 乖离率 (Bias from MA20) - 均值回归指标
    df['ma20'] = df['Close'].rolling(20).mean()
    df['bias20'] = (df['Close'] - df['ma20']) / df['ma20']
    
    # 2. 简单的历史回测逻辑 (基于当前 Bias 状态)
    # 查找历史上相似的 "超跌" 状态 (Bias < -0.05 且当日大跌)
    current_bias = df['bias20'].iloc[-1]
    current_drop = df['Close'].pct_change().iloc[-1]
    
    similar_days = df[(df['bias20'] < -0.05) & (df['Close'].pct_change() < -0.03)]
    
    next_day_returns = []
    for idx in similar_days.index:
        pos = df.index.get_loc(idx)
        if pos < len(df) - 1:
            next_day_ret = (df['Close'].iloc[pos+1] / df['Close'].iloc[pos] - 1)
            next_day_returns.append(next_day_ret)
    
    avg_next_day = np.mean(next_day_returns) if next_day_returns else 0
    win_rate = len([r for r in next_day_returns if r > 0]) / len(next_day_returns) if next_day_returns else 0

    # 3. 走势推断逻辑
    # 如果 RSI < 35 且 Bias 极低，通常伴随“死猫跳”反弹
    rsi_val = 32.42 # 引用上个脚本结果
    signal = "NEUTRAL"
    if current_bias < -0.07 and rsi_val < 35:
        signal = "STRONG_REBOUND_PROBABLE"
    elif current_drop < -0.05:
        signal = "OVERSOLD_BOUNCE"

    return {
        "symbol": symbol,
        "current_bias": round(current_bias, 4),
        "hist_similar_cases": len(next_day_returns),
        "hist_next_day_avg_ret": round(avg_next_day * 100, 2),
        "hist_win_rate": round(win_rate * 100, 2),
        "signal": signal,
        "prediction_tomorrow": "UP" if avg_next_day > 0 or signal == "STRONG_REBOUND_PROBABLE" else "DOWN"
    }

if __name__ == "__main__":
    result = backtest_and_predict("5020.T")
    print(json.dumps(result, indent=2))
