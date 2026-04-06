import sqlite3

conn = sqlite3.connect('japan_market.db')
c = conn.cursor()

# Delete the system auto-generated fill
c.execute("DELETE FROM fills WHERE source='paper_simulator' AND asof='2026-04-06'")
print(f'✅ 已删除 {c.rowcount} 笔由系统自动生成的多余订单。')

# Clear out today's calculated positions and snapshot so the rebuild process processes the manual fills from scratch
c.execute("DELETE FROM positions WHERE asof='2026-04-06'")
c.execute("DELETE FROM account_snapshots WHERE asof='2026-04-06'")

conn.commit()
conn.close()
