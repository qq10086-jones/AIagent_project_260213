import yfinance as yf
from market_db_v1 import MarketDB
from datetime import datetime, timedelta, date
from typing import Dict, List, Tuple


# 1) 初始股票池（你后面可以考虑把这个外置到 json/yaml 或 DB 表里）
TARGET_UNIVERSE: List[Tuple[str, str, str]] = [
    # --- 现有核心：光伏/能源/化学 ---
    ("4063.T", "Shin-Etsu Chemical", "Semicon/Chemical"),
    ("6367.T", "Daikin", "Machinery"),
    ("5020.T", "ENEOS", "Energy"),
    
    # --- 现有核心：半导体/AI ---
    ("8035.T", "Tokyo Electron", "Semicon Equip"),
    ("6857.T", "Advantest", "Semicon Test"),
    ("6146.T", "Disco", "Semicon Process"),
    ("6758.T", "Sony Group", "Tech/Entertainment"), # 新增：索尼（感光元件+娱乐）
    ("6861.T", "Keyence", "Automation"),           # 新增：基恩士（超高利润率，工厂自动化）

    # --- 现有核心：高股息/商社 (巴菲特概念) ---
    ("8058.T", "Mitsubishi Corp", "Trading"),
    ("8001.T", "Itochu", "Trading"),
    ("8031.T", "Mitsui & Co", "Trading"),          # 新增：三井物产（能源资源强）
    ("8002.T", "Marubeni", "Trading"),             # 新增：丸红（农业/电力）

    # --- 金融/保险 (日本加息最大受益者) ---
    ("8306.T", "MUFG", "Bank"),
    ("8316.T", "SMBC", "Bank"),
    ("8766.T", "Tokio Marine", "Insurance"),       # 新增：东京海上（全球顶级财险，非常稳健）
    ("8591.T", "ORIX", "Financial Serv"),          # 新增：欧力士（高股息，业务多元）

    # --- 重工/国防 (地缘政治对冲) ---
    ("7011.T", "Mitsubishi Heavy", "Defense/Space"), # 新增：三菱重工（国防、核能、燃气轮机）
    ("7012.T", "Kawasaki Heavy", "Machinery"),       # 新增：川崎重工（液氢运输、摩托、机器人）

    # --- 汽车/运输 (出口与汇率敏感) ---
    ("7203.T", "Toyota Motor", "Auto"),              # 新增：丰田（日本市值的定海神针）
    ("9101.T", "NYK Line", "Shipping"),              # 新增：日本邮船（航运周期股，高波动高分红）

    # --- 消费/内需 (防御性板块) ---
    ("9432.T", "NTT", "Telecom"),
    ("2914.T", "JT", "Tobacco"),
    ("9983.T", "Fast Retailing", "Retail"),          # 新增：优衣库母公司（日经225权重第一，影响指数极大）
    ("7974.T", "Nintendo", "Gaming"),                # 新增：任天堂（拥有最强IP，且现金流充裕）
    ("4661.T", "Oriental Land", "Leisure"),          # 新增：迪士尼运营方（日本最强旅游/体验经济）

    # --- 基准 ---
    ("1321.T", "Nikkei 225 ETF", "Benchmark"),
    ("1570.T", "Nikkei Lev", "Benchmark_2x"),        # 新增：日经2倍杠杆（用于观察高beta情绪，不一定交易）
]


def _date_to_str(d: date) -> str:
    return d.strftime("%Y-%m-%d")


def update_database(db_path: str = "japan_market.db", default_lookback_days: int = 730) -> None:
    """
    优化点：
      1) 增量更新：每个 symbol 只拉取库里最新日期之后的数据
      2) 仍然用一次批量 download 来减少网络开销（按最早 start 拉一遍，然后对每个 symbol 切片）
      3) 输出更清晰的日志，失败时打印异常原因
    """
    print("🚀 启动自动化数据收集系统...")
    db = MarketDB(db_path)

    # 1) 更新 tickers 基础信息
    print("📋 更新股票基础信息...")
    now = datetime.now()
    formatted_tickers = [(sym, name, sector, "Auto-Added", now) for sym, name, sector in TARGET_UNIVERSE]
    db.save_tickers(formatted_tickers)

    tickers = [t[0] for t in TARGET_UNIVERSE]

    # 2) 计算每个 symbol 的增量 start
    latest_map: Dict[str, date] = {sym: db.get_latest_date(sym) for sym in tickers}
    today = date.today()

    # 若 DB 为空，默认回溯两年（default_lookback_days）
    start_map: Dict[str, date] = {}
    for sym in tickers:
        last = latest_map[sym]
        if last is None:
            start_map[sym] = today - timedelta(days=default_lookback_days)
        else:
            start_map[sym] = last + timedelta(days=1)

    # 判断是否需要更新
    need_update = [sym for sym in tickers if start_map[sym] <= today]
    if not need_update:
        print("✅ 所有标的都已是最新，无需更新。")
        db.close()
        return

    # 3) 批量下载：按“最早 start”统一拉取一次，再对每个 symbol 做增量切片
    earliest_start = min(start_map[sym] for sym in need_update)
    print(
        f"📥 开始下载 {len(tickers)} 只标的的历史数据 "
        f"(start={_date_to_str(earliest_start)} -> end={_date_to_str(today)}) ..."
    )

    # NOTE:
    # - auto_adjust=True => OHLC 都是复权口径（策略/回测更方便）
    # - group_by='ticker' 便于取每只 ticker 的 df
    data = yf.download(
        tickers,
        start=_date_to_str(earliest_start),
        end=_date_to_str(today + timedelta(days=1)),  # yfinance end 是“开区间”，+1天更稳
        group_by="ticker",
        auto_adjust=True,
        threads=True,
        progress=False,
    )

    # 记录口径到 meta（可选）
    try:
        db.set_meta("price_mode", "yfinance:auto_adjust=True")
        db.set_meta("last_update_run", datetime.now().isoformat(timespec="seconds"))
    except Exception:
        pass

    # 4) 写库（增量切片）
    print("💾 正在写入数据库...")
    total_rows = 0
    for sym in tickers:
        try:
            # yfinance：单 ticker 时 data 结构不同；这里统一处理
            df = data if len(tickers) == 1 else data.get(sym)

            if df is None or df.empty:
                print(f"⚠️ {sym}: 无数据返回（可能是停牌/代码问题/网络波动）")
                continue

            # 丢掉全空行（比如 volume 全空）
            df = df.dropna(how="all")
            if df.empty:
                print(f"⚠️ {sym}: 数据全为空行")
                continue

            # 增量切片：只保留 start_map 之后的部分
            start_d = start_map[sym]
            df = df.loc[df.index.date >= start_d]

            if df.empty:
                print(f"✅ {sym}: 无新增交易日数据")
                continue

            rows = db.save_prices(sym, df)
            total_rows += rows

        except Exception as e:
            print(f"❌ {sym}: 更新失败 -> {type(e).__name__}: {e}")

    db.close()
    print(f"✅ 系统任务完成！共写入/更新 {total_rows} 行K线数据。数据库：{db_path}")


if __name__ == "__main__":
    update_database()
