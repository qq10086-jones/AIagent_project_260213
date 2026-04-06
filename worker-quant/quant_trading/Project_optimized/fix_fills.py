import sqlite3
import datetime

db_path = 'japan_market.db'
conn = sqlite3.connect(db_path)
c = conn.cursor()

# 1. 查找并删除由系统自动买入的那 200 股 4005.T（无 external_ref 备注的记录）
today_str = "2026-04-06%"
c.execute("""
    DELETE FROM fills 
    WHERE symbol='4005.T' 
    AND external_ref IS NULL 
    AND ts LIKE ?
""", (today_str,))
deleted_rows = c.rowcount
conn.commit()
print(f"✅ 已删除 {deleted_rows} 笔由系统自动生成的多余订单。")

# 2. 我们还要去 orders 表里把那笔自动生成的指令状态设为 expired，防止再次执行
c.execute("""
    UPDATE orders 
    SET status = 'expired' 
    WHERE symbol='4005.T' 
    AND asof='2026-04-06'
""")
conn.commit()

# 3. 删除错乱的 account_snapshot，让接下来我们可以重算
c.execute("DELETE FROM account_snapshots WHERE asof='2026-04-06'")
conn.commit()
conn.close()
print("✅ 数据清理完成。")
