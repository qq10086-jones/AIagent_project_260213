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

# Default universe (same as your v1)
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
