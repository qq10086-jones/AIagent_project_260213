
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta

class NexusAdvancedStrategyEngine:
    def __init__(self, capital, risk_tolerance="MEDIUM"):
        self.capital = capital
        self.risk_tolerance = risk_tolerance
        # 风险参数映射
        self.risk_params = {
            "LOW": {"atr_sl": 1.5, "atr_tp": 2.0, "max_pos_pct": 0.2},
            "MEDIUM": {"atr_sl": 2.0, "atr_tp": 3.0, "max_pos_pct": 0.4},
            "HIGH": {"atr_sl": 3.0, "atr_tp": 5.0, "max_pos_pct": 0.8}
        }[risk_tolerance]

    # ==========================================
    # 1. 数学与风控模块 (Math & Risk Control)
    # ==========================================
    
    def calculate_atr(self, df: pd.DataFrame, window=14):
        """计算真实波动幅度 (ATR) - 用于动态止损"""
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        return true_range.rolling(window).mean().iloc[-1]

    def market_regime_filter(self, index_symbol="^N225"):
        """大盘状态过滤器 (隐式马尔可夫模型的基础平替)"""
        try:
            df = yf.Ticker(index_symbol).history(period="6mo")
            ma20 = df['Close'].rolling(20).mean().iloc[-1]
            ma50 = df['Close'].rolling(50).mean().iloc[-1]
            
            if df['Close'].iloc[-1] > ma20 and ma20 > ma50:
                return "BULL" # 均线多头排列，支持进攻
            elif df['Close'].iloc[-1] < ma20 and ma20 < ma50:
                return "BEAR" # 均线空头排列，必须收缩防线
            else:
                return "CHOPPY" # 震荡市，适合均值回归
        except:
            return "UNKNOWN"

    def dynamic_position_sizing(self, stock_price, atr, available_cash):
        """基于 ATR 的风险平价仓位计算 (Risk Parity)"""
        # 假设单笔交易我们愿意承担总资金 1% 的风险
        risk_per_trade = self.capital * 0.01 
        # 止损幅度即为风险幅度
        stop_loss_dist = atr * self.risk_params["atr_sl"]
        if stop_loss_dist == 0: return 0
        
        # 理想股数 = 愿意承担的资金风险 / 每股止损风险
        ideal_shares = risk_per_trade / stop_loss_dist
        
        # 日本市场必须是 100 的整数倍
        ideal_lots = int(ideal_shares // 100)
        
        # 资金约束与最大持仓限制
        max_lots_by_cash = int((available_cash * self.risk_params["max_pos_pct"]) / (stock_price * 100))
        
        final_lots = min(ideal_lots, max_lots_by_cash)
        return final_lots * 100

    # ==========================================
    # 2. 情报分析与历史回归 (NLP & Historical Event Regression)
    # ==========================================
    
    def analyze_news_impact(self, news_text, llm_analysis_func=None):
        """
        [概念演示] 将外部新闻输入LLM，提取事件标签，并在历史数据库中进行概率回归。
        由于本地无海量历史标签库，这里用结构展示其逻辑。
        """
        # 1. LLM 提取标签
        # tags = llm_analysis_func(news_text) 
        # example tags: {"entity": "BOJ", "event": "Rate Hike", "sentiment": -0.8}
        tags = {"event": "Market Crash", "sentiment": -0.8}
        
        # 2. 模拟从 SQLite (db_update.py 维护的库) 中查询历史相似事件的 5日胜率
        # 假设历史回归表明：大盘暴跌日(Sentiment < -0.5) 后买入高息股的5日胜率是 65%
        historical_win_rate = 0.65 
        expected_return_5d = 0.02 # 2%
        
        return {
            "tags": tags,
            "historical_win_rate": historical_win_rate,
            "expected_return_5d": expected_return_5d,
            "event_multiplier": 1.5 if historical_win_rate > 0.6 else 0.5
        }

    # ==========================================
    # 3. 终极决策输出
    # ==========================================
    
    def evaluate_candidate(self, symbol, current_cash, news_context=""):
        """综合评估一只股票是否值得买入，并给出明确的止盈止损线"""
        # 1. 大盘环境评估
        regime = self.market_regime_filter()
        if regime == "BEAR":
            return {"signal": "REJECT", "reason": "大盘处于下降通道，拒绝接飞刀。"}
            
        # 2. 个股技术面与 ATR
        df = yf.Ticker(symbol).history(period="3mo")
        if df.empty or len(df) < 30:
            return {"signal": "REJECT", "reason": "数据不足"}
            
        price = df['Close'].iloc[-1]
        atr = self.calculate_atr(df)
        
        # 3. 反转确认 (Momentum Reversal)
        # 不买正在暴跌的阴线，必须是阳线（收盘 > 开盘）且最好在下轨
        is_green_candle = df['Close'].iloc[-1] > df['Open'].iloc[-1]
        
        if not is_green_candle and regime == "CHOPPY":
            return {"signal": "WAIT", "reason": "未见反转信号(非阳线)，继续等待。"}
            
        # 4. 新闻情报事件加成
        event_impact = self.analyze_news_impact(news_context)
        
        # 5. 计算仓位与止盈止损
        shares_to_buy = self.dynamic_position_sizing(price, atr, current_cash)
        
        if shares_to_buy < 100:
            return {"signal": "REJECT", "reason": f"资金风险配比不允许买入哪怕1手。ATR: {atr:.2f}"}
            
        stop_loss = price - (atr * self.risk_params["atr_sl"])
        take_profit = price + (atr * self.risk_params["atr_tp"])
        
        return {
            "signal": "BUY",
            "reason": f"大盘状态: {regime}, 事件胜率: {event_impact['historical_win_rate']*100}%, 出现企稳信号。",
            "action": {
                "symbol": symbol,
                "shares": shares_to_buy,
                "current_price": round(price, 2),
                "stop_loss": round(stop_loss, 2),
                "take_profit": round(take_profit, 2)
            },
            "metrics": {
                "regime": regime,
                "atr": round(float(atr), 4),
                "historical_win_rate": round(float(event_impact["historical_win_rate"]), 4),
                "expected_return_5d": round(float(event_impact["expected_return_5d"]), 4),
            },
        }

if __name__ == "__main__":
    # 模拟环境测试
    engine = NexusAdvancedStrategyEngine(capital=400000, risk_tolerance="MEDIUM")
    
    print("🧠 Nexus 进阶策略引擎测试启动...")
    print(f"📊 当前大盘环境: {engine.market_regime_filter()}")
    
    # 假设手里有 27w 现金，评估 5020.T (昨日暴跌，需等待企稳信号)
    res = engine.evaluate_candidate("5020.T", current_cash=271300, news_context="油价波动")
    print(f"\n对 5020.T 的决策:\n{res}")
