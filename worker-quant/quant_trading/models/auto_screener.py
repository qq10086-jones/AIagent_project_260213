import yfinance as yf
from market_db import MarketDB
from datetime import datetime

# 1. 定义我们关注的初始股票池 (结合你的光伏、科技背景和SBI高股息需求)
TARGET_UNIVERSE = [
    # --- 你的专业领域：光伏/能源/化学 ---
    ('4063.T', 'Shin-Etsu Chemical', 'Semicon/Chemical'), # 信越化学 (硅片霸主)
    ('6367.T', 'Daikin', 'Machinery'),
    ('5020.T', 'ENEOS', 'Energy'),
    
    # --- 你的兴趣：半导体/AI (配合 AMD 7900XTX 的逻辑) ---
    ('8035.T', 'Tokyo Electron', 'Semicon Equip'), # 东京电子
    ('6857.T', 'Advantest', 'Semicon Test'),       # 爱德万
    ('6146.T', 'Disco', 'Semicon Process'),        # Disco
    
    # --- 高股息/稳健 (适合 SBI 长期持有) ---
    ('9432.T', 'NTT', 'Telecom'),       # 电信
    ('2914.T', 'JT', 'Tobacco'),        # 日本烟草
    ('8306.T', 'MUFG', 'Bank'),         # 三菱UFJ
    ('8316.T', 'SMBC', 'Bank'),         # 三井住友
    ('8058.T', 'Mitsubishi Corp', 'Trading'), # 三菱商事
    ('8001.T', 'Itochu', 'Trading'),          # 伊藤忠
    
    # --- 基准 ---
    ('1321.T', 'Nikkei 225 ETF', 'Benchmark')
]

def update_database():
    print("🚀 启动自动化数据收集系统...")
    db = MarketDB() # 连接数据库
    
    # 1. 更新股票列表信息
    print("📋 更新股票基础信息...")
    formatted_tickers = []
    now = datetime.now()
    for t in TARGET_UNIVERSE:
        # 格式: (symbol, name, sector, memo, last_updated)
        formatted_tickers.append((t[0], t[1], t[2], "Auto-Added", now))
    db.save_tickers(formatted_tickers)
    
    # 2. 批量下载数据 (使用 yfinance 的多线程下载)
    ticker_list = [t[0] for t in TARGET_UNIVERSE]
    print(f"📥 开始下载 {len(ticker_list)} 只股票的历史数据 (过去2年)...")
    
    # group_by='ticker' 方便后续处理
    data = yf.download(ticker_list, period="2y", group_by='ticker', auto_adjust=True, threads=True)
    
    # 3. 存入数据库
    print("💾 正在写入数据库...")
    for symbol in ticker_list:
        try:
            # 提取单只股票的 DataFrame
            df = data[symbol].copy()
            # 剔除空值
            df.dropna(inplace=True)
            if not df.empty:
                db.save_prices(symbol, df)
        except Exception as e:
            print(f"⚠️ 跳过 {symbol}: 数据获取异常")
            
    db.close()
    print("✅ 系统任务完成！数据已更新至 japan_market.db")

if __name__ == "__main__":
    update_database()