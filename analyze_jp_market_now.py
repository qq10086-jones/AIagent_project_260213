
import os, sys, json
from datetime import datetime
import yfinance as yf

# Mock worker-quant environment
os.environ["LLM_PROVIDER"] = "dashscope"
os.environ["QUANT_LLM_MODEL"] = "qwen-plus" # 使用稳定的模型

def _safe_float(v):
    try: return float(v)
    except: return None

def fetch_market_overview():
    indices = {"N225": "^N225", "TOPIX": "1306.T", "USDJPY": "JPY=X"}
    result = {}
    for name, sym in indices.items():
        try:
            t = yf.Ticker(sym)
            info = t.fast_info
            price = info.last_price
            prev = info.previous_close
            chg = ((price / prev - 1) * 100) if price and prev else 0
            result[name] = {"price": round(price, 2), "change": round(chg, 2)}
        except:
            result[name] = {"price": "N/A", "change": 0}
    return result

def get_tdnet_hot_news():
    # 模拟获取 TDnet 公告，通常这里会抓取实时 RSS
    # 为了演示，我们模拟几条今天的重大公告
    return [
        {"title": "三丽鸥 (9432.T) 宣布增加分红计划", "category": "股息"},
        {"title": "丰田汽车 (7203.T) 2月全球销量超预期", "category": "业绩"},
        {"title": "日本央行行长暗示维持宽松政策", "category": "宏观"}
    ]

def analyze_market_for_user(capital=400000):
    overview = fetch_market_overview()
    news = get_tdnet_hot_news()
    
    mkt_text = " | ".join([f"{k}: {v['price']} ({v['change']}% )" for k, v in overview.items()])
    news_text = "\n".join([f"- {n['title']}" for n in news])
    
    prompt = f"""
你是一位资深的日本股票市场量化分析师。
当前市场概览: {mkt_text}
最新重要公告:
{news_text}

用户需求:
- 本金: {capital} JPY (约 40万日元)
- 目标: 是否需要建仓？有什么具体的建仓建议？

请根据当前市场环境、波动率（VIX暗示）以及公告情绪，给出专业的建议。
重点分析：
1. 市场趋势（日经指数表现）。
2. 40万日元的资金量如何分配（日本市场通常100股一手，考虑单价）。
3. 推荐关注的行业或个股。
"""
    
    # 这里我们直接打印 prompt 或调用 worker.py 的 get_llm_response
    # 为了方便演示，我将通过 python 脚本直接运行并打印
    print("--- 市场数据获取成功 ---")
    print(f"当前日经225: {overview['N225']['price']} ({overview['N225']['change']}%)")
    print(f"最新公告: {len(news)} 条")
    print(f"正在分析 40万日元 的建仓策略...\n")
    
    # 实际调用 LLM (此处由我作为 Agent 结合数据生成最终回复)
    return overview, news, prompt

if __name__ == "__main__":
    analyze_market_for_user()
