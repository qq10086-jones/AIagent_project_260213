import sqlite3
import pandas as pd
from datetime import datetime

class MarketDB:
    def __init__(self, db_path="japan_market.db"):
        """初始化数据库连接，文件会自动创建在当前目录下"""
        self.conn = sqlite3.connect(db_path)
        self.create_tables()

    def create_tables(self):
        """创建基础表结构：股票列表 + 日线行情"""
        with self.conn:
            # 1. 股票基础信息表 (Tickers)
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS tickers (
                    symbol TEXT PRIMARY KEY,
                    name TEXT,
                    sector TEXT,
                    memo TEXT,
                    last_updated TIMESTAMP
                )
            """)
            # 2. 日线行情表 (Prices) - 包含必要的OHLCV
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS daily_prices (
                    symbol TEXT,
                    date TEXT,
                    open REAL,
                    high REAL,
                    low REAL,
                    close REAL,
                    volume REAL,
                    PRIMARY KEY (symbol, date)
                )
            """)
            # 创建索引以加速查询
            self.conn.execute("CREATE INDEX IF NOT EXISTS idx_date ON daily_prices (date)")

    def save_tickers(self, ticker_list):
        """批量保存/更新股票代码信息"""
        # ticker_list 格式: [('9432.T', 'NTT', 'Comm'), ...]
        with self.conn:
            self.conn.executemany("""
                INSERT OR REPLACE INTO tickers (symbol, name, sector, memo, last_updated)
                VALUES (?, ?, ?, ?, ?)
            """, ticker_list)
        print(f"✅ 已更新 {len(ticker_list)} 只股票的基础信息")

    def save_prices(self, symbol, df):
        """保存单只股票的历史数据 (适配 yfinance 格式)"""
        if df.empty:
            return
        
        # 清洗数据，确保格式统一
        data = df.reset_index().copy()
        # 处理 yfinance 可能不同的列名
        data.columns = [c.lower() for c in data.columns]
        if 'date' not in data.columns: 
             # 假如索引是日期但列名没对上
            data.rename(columns={'index': 'date'}, inplace=True)
            
        # 转换日期格式为字符串 YYYY-MM-DD
        data['date'] = data['date'].dt.strftime('%Y-%m-%d')
        data['symbol'] = symbol
        
        # 选取需要的列
        cols = ['symbol', 'date', 'open', 'high', 'low', 'close', 'volume']
        records = data[cols].to_records(index=False).tolist()

        try:
            with self.conn:
                self.conn.executemany("""
                    INSERT OR REPLACE INTO daily_prices (symbol, date, open, high, low, close, volume)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, records)
            print(f"📈 {symbol}: 已存储 {len(records)} 条K线数据")
        except Exception as e:
            print(f"❌ {symbol} 存储失败: {e}")

    def get_price_df(self, symbol):
        """读取数据给 ss6.py 使用"""
        query = "SELECT date, open, high, low, close, volume FROM daily_prices WHERE symbol = ? ORDER BY date"
        df = pd.read_sql(query, self.conn, params=(symbol,), index_col='date', parse_dates=['date'])
        return df

    def close(self):
        self.conn.close()

if __name__ == "__main__":
    # 测试代码
    db = MarketDB()
    print("数据库已初始化完成：japan_market.db")
    db.close()