"""DB updater: SQLite <- yfinance (auto_adjust)

This is a refactor of auto_screener_v1.py:
- Keeps the same TARGET_UNIVERSE default list.
- Adds optional universe loading from JSON/YAML.
- Writes meta keys for audit.

Usage:
  python db_update.py --db japan_market.db
  python db_update.py --db japan_market.db --universe universe.json
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta, date
from typing import Dict, List, Tuple, Optional

import yfinance as yf

from market_db_v2 import MarketDB

# Default universe - TSE Prime前300流动性股票（扩展至约100只）
# 筛选标准：流动性好、覆盖多行业、含较多1手成本<15万JPY的品种（由screener进一步过滤）
TARGET_UNIVERSE: List[Tuple[str, str, str]] = [
    # --- 半导体/AI/精密仪器 ---
    ("4063.T", "Shin-Etsu Chemical", "Semicon/Chemical"),
    ("8035.T", "Tokyo Electron", "Semicon Equip"),
    ("6857.T", "Advantest", "Semicon Test"),
    ("6146.T", "Disco", "Semicon Process"),
    ("6861.T", "Keyence", "Automation"),
    ("6723.T", "Renesas Electronics", "Semicon"),
    ("7735.T", "Screen Holdings", "Semicon Equip"),
    ("6920.T", "Lasertec", "Semicon Equip"),

    # --- 电子/电机/机械 ---
    ("6758.T", "Sony Group", "Tech/Entertainment"),
    ("6501.T", "Hitachi", "Conglomerate"),
    ("6503.T", "Mitsubishi Electric", "Electrical Equip"),
    ("6702.T", "Fujitsu", "IT Services"),
    ("6752.T", "Panasonic Holdings", "Electronics"),
    ("6367.T", "Daikin", "Machinery"),
    ("6301.T", "Komatsu", "Construction Mach"),
    ("6326.T", "Kubota", "Machinery"),
    ("6506.T", "Yaskawa Electric", "Robotics"),
    ("6988.T", "Nitto Denko", "Materials"),
    ("4901.T", "Fujifilm Holdings", "Imaging/Healthcare"),
    ("7751.T", "Canon", "Electronics"),
    ("7733.T", "Olympus", "Medical Devices"),
    ("6471.T", "NSK", "Bearings"),
    ("6770.T", "Alps Alpine", "Electronic Parts"),

    # --- 重工/国防/航天 ---
    ("7011.T", "Mitsubishi Heavy", "Defense/Space"),
    ("7012.T", "Kawasaki Heavy", "Machinery"),
    ("1963.T", "JGC Holdings", "Engineering"),

    # --- 商社/贸易 (巴菲特概念) ---
    ("8058.T", "Mitsubishi Corp", "Trading"),
    ("8001.T", "Itochu", "Trading"),
    ("8031.T", "Mitsui & Co", "Trading"),
    ("8002.T", "Marubeni", "Trading"),
    ("8053.T", "Sumitomo Corp", "Trading"),

    # --- 金融/银行/保险/证券 ---
    ("8306.T", "MUFG", "Bank"),
    ("8316.T", "SMBC", "Bank"),
    ("8411.T", "Mizuho FG", "Bank"),
    ("8766.T", "Tokio Marine", "Insurance"),
    ("8591.T", "ORIX", "Financial Serv"),
    ("8604.T", "Nomura Holdings", "Securities"),
    ("8630.T", "Sompo Holdings", "Insurance"),
    ("8725.T", "MS&AD Insurance", "Insurance"),
    ("8697.T", "Japan Exchange Group", "Exchange"),
    ("8309.T", "SMTB", "Financial"),

    # --- 汽车/零部件 (出口敏感) ---
    ("7203.T", "Toyota Motor", "Auto"),
    ("7267.T", "Honda Motor", "Auto"),
    ("7201.T", "Nissan Motor", "Auto"),
    ("7269.T", "Suzuki Motor", "Auto"),
    ("7261.T", "Mazda Motor", "Auto"),
    ("7270.T", "Subaru", "Auto"),
    ("7211.T", "Mitsubishi Motors", "Auto"),

    # --- 运输/物流/航运 ---
    ("9101.T", "NYK Line", "Shipping"),
    ("9104.T", "Mitsui OSK Lines", "Shipping"),
    ("9107.T", "Kawasaki Kisen", "Shipping"),
    ("9020.T", "JR East", "Railway"),
    ("9201.T", "Japan Airlines", "Aviation"),
    ("9202.T", "ANA Holdings", "Aviation"),

    # --- 能源/化工 ---
    ("5020.T", "ENEOS", "Energy"),
    ("1605.T", "Inpex", "Oil & Gas"),
    ("4005.T", "Sumitomo Chemical", "Chemical"),
    ("4183.T", "Mitsui Chemicals", "Chemical"),
    ("5713.T", "Sumitomo Metal Mining", "Mining"),
    ("5714.T", "Dowa Holdings", "Metals"),

    # --- 钢铁/基础材料 ---
    ("5401.T", "Nippon Steel", "Steel"),
    ("5411.T", "JFE Holdings", "Steel"),

    # --- 电力/公用事业 (防御性) ---
    ("9501.T", "Tokyo Electric Power", "Utility"),
    ("9502.T", "Chubu Electric Power", "Utility"),
    ("9503.T", "Kansai Electric Power", "Utility"),

    # --- 电信 ---
    ("9432.T", "NTT", "Telecom"),
    ("9433.T", "KDDI", "Telecom"),
    ("9434.T", "SoftBank Corp", "Telecom"),

    # --- 消费/食品/饮料/日用品 ---
    ("2914.T", "JT", "Tobacco"),
    ("2502.T", "Asahi Group Holdings", "Beverage"),
    ("2503.T", "Kirin Holdings", "Beverage"),
    ("2801.T", "Kikkoman", "Food"),
    ("2282.T", "Nippon Ham", "Food"),
    ("4452.T", "Kao", "Consumer Goods"),

    # --- 零售 ---
    ("9983.T", "Fast Retailing", "Retail"),
    ("3382.T", "Seven & i Holdings", "Retail"),
    ("8267.T", "Aeon", "Retail"),

    # --- 不动产 ---
    ("8801.T", "Mitsui Fudosan", "Real Estate"),
    ("8802.T", "Mitsubishi Estate", "Real Estate"),
    ("8830.T", "Sumitomo Realty", "Real Estate"),

    # --- 制药/医疗 ---
    ("4502.T", "Takeda Pharmaceutical", "Pharma"),
    ("4503.T", "Astellas Pharma", "Pharma"),
    ("4568.T", "Daiichi Sankyo", "Pharma"),
    ("4523.T", "Eisai", "Pharma"),
    ("4519.T", "Chugai Pharmaceutical", "Pharma"),
    ("4578.T", "Ono Pharmaceutical", "Pharma"),

    # --- 互联网/平台/娱乐 ---
    ("4755.T", "Rakuten Group", "Internet"),
    ("6098.T", "Recruit Holdings", "HR/Internet"),
    ("3659.T", "Nexon", "Gaming"),
    ("7974.T", "Nintendo", "Gaming"),
    ("7832.T", "Bandai Namco", "Entertainment"),
    ("4661.T", "Oriental Land", "Leisure"),
    ("2413.T", "M3 Inc", "Medical Internet"),

    # --- 基准（不参与交易，仅供benchmark对比和screener数据收集） ---
    ("1321.T", "Nikkei 225 ETF", "Benchmark"),
    ("1570.T", "Nikkei Lev", "Benchmark_2x"),
    ("1306.T", "TOPIX ETF", "Benchmark_TOPIX"),
]

def _date_to_str(d: date) -> str:
    return d.strftime("%Y-%m-%d")

def load_universe(path: str) -> List[Tuple[str, str, str]]:
    if not path:
        return TARGET_UNIVERSE
    p = path.lower()
    if p.endswith(".json"):
        obj = json.loads(open(path, "r", encoding="utf-8").read())
    elif p.endswith((".yml", ".yaml")):
        try:
            import yaml  # type: ignore
        except Exception as e:
            raise RuntimeError("PyYAML not installed. Either install pyyaml or use a JSON universe file.") from e
        obj = yaml.safe_load(open(path, "r", encoding="utf-8"))
    else:
        raise ValueError("Universe file must be .json or .yaml/.yml")

    # Accept formats:
    # 1) [{"symbol":"4063.T","name":"...","sector":"..."}, ...]
    # 2) [["4063.T","name","sector"], ...]
    out: List[Tuple[str, str, str]] = []
    for it in obj:
        if isinstance(it, dict):
            out.append((it["symbol"], it.get("name",""), it.get("sector","")))
        else:
            out.append((it[0], it[1], it[2] if len(it) > 2 else ""))
    return out

def update_database(db_path: str = "japan_market.db", default_lookback_days: int = 730, universe_path: Optional[str]=None) -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass
    print("DB updater: start")
    db = MarketDB(db_path)

    universe = load_universe(universe_path) if universe_path else TARGET_UNIVERSE

    # 1) Update ticker metadata
    now = datetime.now()
    formatted = [(sym, name, sector, "Auto-Added", now) for sym, name, sector in universe]
    db.save_tickers(formatted)

    tickers = [t[0] for t in universe]

    # 2) Incremental start dates
    latest_map: Dict[str, date] = {sym: db.get_latest_date(sym) for sym in tickers}
    today = date.today()

    start_map: Dict[str, date] = {}
    for sym in tickers:
        last = latest_map[sym]
        start_map[sym] = (today - timedelta(days=default_lookback_days)) if last is None else (last + timedelta(days=1))

    need = [sym for sym in tickers if start_map[sym] <= today]
    if not need:
        print("No updates needed.")
        db.close()
        return

    earliest = min(start_map[sym] for sym in need)
    print(f"Downloading {len(tickers)} tickers: {earliest} -> {today}")

    data = yf.download(
        tickers,
        start=_date_to_str(earliest),
        end=_date_to_str(today + timedelta(days=1)),
        group_by="ticker",
        auto_adjust=True,
        threads=True,
        progress=False,
    )

    # audit metadata
    try:
        db.set_meta("price_mode", "yfinance:auto_adjust=True")
        db.set_meta("last_update_run", datetime.now().isoformat(timespec="seconds"))
        db.set_meta("universe_size", str(len(tickers)))
    except Exception:
        pass

    total_rows = 0
    for sym in tickers:
        try:
            df = data if len(tickers) == 1 else data.get(sym)
            if df is None or df.empty:
                print(f"WARN {sym}: no data")
                continue
            df = df.dropna(how="all")
            if df.empty:
                print(f"WARN {sym}: all-NA")
                continue
            df = df.loc[df.index.date >= start_map[sym]]
            if df.empty:
                print(f"{sym}: no new rows")
                continue
            rows = db.save_prices(sym, df)
            total_rows += rows
        except Exception as e:
            print(f"ERROR {sym}: {type(e).__name__}: {e}")

    db.close()
    print(f"Done. Rows upserted: {total_rows}. DB={db_path}")

def _extract_default_universe_tuple_list(src: str) -> str:
    # Pull TARGET_UNIVERSE from original file text (best effort) to preserve your list
    m = re.search(r"TARGET_UNIVERSE\s*:\s*List\[Tuple\[str,\s*str,\s*str\]\]\s*=\s*(\[.*?\])\n\n", src, re.S)
    if not m:
        return "[]"
    return m.group(1)

if __name__ == "__main__":
    import re
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="japan_market.db")
    ap.add_argument("--lookback", type=int, default=730)
    ap.add_argument("--universe", default=None, help="optional universe file: JSON/YAML")
    args = ap.parse_args()
    update_database(args.db, args.lookback, args.universe)
