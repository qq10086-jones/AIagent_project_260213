"""动态 Universe 构建器 — 从 TSE 全量上市公司中筛选交易候选池。

数据源优先级:
  1. JPX 官网 CSV (东証上場銘柄一覧) — 最权威
  2. yfinance 批量验证 — 检查数据可用性
  3. 硬编码 seed (db_update.py TARGET_UNIVERSE) — fallback

筛选逻辑:
  Phase 1: 获取 TSE 全量代码
  Phase 2: 排除 ETF/REIT/优先股等非普通股
  Phase 3: yfinance 数据可用性验证 + 流动性过滤
  Phase 4: 输出 universe.json

Usage:
  python universe_builder.py --config config.yaml
  python universe_builder.py --output universe.json --min-adv 10000000
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone, timedelta, date
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

# ── TSE 代码范围 ────────────────────────────────────────────────────
# 东证上场铭柄: 4桁代码 1000-9999 (.T suffix)
# 一部の銘柄は 5桁 (10000-99999) だが数が少ない
TSE_CODE_RANGES = [
    (1300, 1999),   # ETF/REIT 帯 — フィルタで除外
    (2000, 9999),   # メイン帯
]

# 排除パターン: ETF, REIT, 優先株, インフラファンド, 受益証券
EXCLUDE_SECTORS = {
    "Benchmark", "Benchmark_2x", "Benchmark_TOPIX", "Benchmark_VIX",
}

# Benchmark ETF は universe に入れるがトレード対象外（screener が除外）
BENCHMARK_TICKERS = [
    ("1321.T", "Nikkei 225 ETF", "Benchmark"),
    ("1570.T", "Nikkei Lev", "Benchmark_2x"),
    ("1306.T", "TOPIX ETF", "Benchmark_TOPIX"),
    ("1552.T", "VIX Short-Term Futures ETF", "Benchmark_VIX"),
]


def _jpx_csv_url() -> str:
    """JPX 公式の上場銘柄一覧 CSV URL (Excel ベース, 毎月更新)。"""
    return "https://www.jpx.co.jp/markets/statistics-equities/misc/tvdivq0000001vg2-att/data_j.xls"


def fetch_tse_from_jpx() -> list[dict] | None:
    """JPX 公式リストからTSE上場銘柄を取得。失敗時 None。"""
    try:
        df = pd.read_excel(_jpx_csv_url(), dtype=str)
        # カラム名は日本語: コード, 銘柄名, 市場・商品区分, 33業種コード, 33業種区分, ...
        code_col = [c for c in df.columns if "コード" in str(c)]
        name_col = [c for c in df.columns if "銘柄名" in str(c)]
        market_col = [c for c in df.columns if "市場" in str(c)]
        sector_col = [c for c in df.columns if "業種" in str(c) and "区分" in str(c)]

        if not code_col:
            print("[universe] JPX CSV: コード列が見つかりません", file=sys.stderr)
            return None

        results = []
        for _, row in df.iterrows():
            code = str(row[code_col[0]]).strip()
            if not code.isdigit():
                continue
            symbol = f"{code}.T"
            name = str(row[name_col[0]]).strip() if name_col else ""
            market = str(row[market_col[0]]).strip() if market_col else ""
            sector = str(row[sector_col[0]]).strip() if sector_col else ""

            # 市場フィルタ: プライム/スタンダード/グロースのみ
            market_lower = market.lower()
            if not any(k in market_lower for k in ["プライム", "スタンダード", "グロース",
                                                     "prime", "standard", "growth"]):
                continue

            results.append({
                "symbol": symbol,
                "name": name,
                "sector": sector,
                "market": market,
                "source": "jpx",
            })
        print(f"[universe] JPX CSV: {len(results)} 銘柄取得")
        return results
    except Exception as e:
        print(f"[universe] JPX CSV 取得失敗: {e}", file=sys.stderr)
        return None


def generate_tse_codes() -> list[str]:
    """TSE 4桁コード候補を生成 (2000-9999)。"""
    codes = []
    for start, end in TSE_CODE_RANGES:
        if start >= 2000:  # ETF帯 (1xxx) はスキップ — benchmark は別途追加
            for i in range(start, end + 1):
                codes.append(f"{i}.T")
    return codes


def verify_yfinance_batch(
    symbols: list[str],
    batch_size: int = 200,
    min_adv: float = 10_000_000,
    min_days: int = 30,
    max_cost_per_lot: float = 500_000,
    lot_size: int = 100,
) -> list[dict]:
    """yfinance でデータ可用性・流動性を一括検証。"""
    import yfinance as yf

    alive = []
    total = len(symbols)
    for i in range(0, total, batch_size):
        batch = symbols[i:i + batch_size]
        batch_num = i // batch_size + 1
        total_batches = (total + batch_size - 1) // batch_size
        print(f"[universe] yfinance 検証 batch {batch_num}/{total_batches} ({len(batch)} 銘柄)...")

        try:
            data = yf.download(
                batch,
                period="60d",
                group_by="ticker",
                auto_adjust=True,
                threads=True,
                progress=False,
            )
        except Exception as e:
            print(f"[universe] yfinance batch {batch_num} 失敗: {e}", file=sys.stderr)
            continue

        if data is None or data.empty:
            continue

        for sym in batch:
            try:
                if len(batch) == 1:
                    df = data
                else:
                    df = data[sym] if sym in data.columns.get_level_values(0) else None

                if df is None or df.empty:
                    continue

                close = df["Close"].dropna()
                volume = df["Volume"].dropna()

                if len(close) < min_days:
                    continue

                # 流動性チェック: 20日ADV
                adv_series = (close * volume).tail(20)
                adv = float(adv_series.mean()) if len(adv_series) >= 10 else 0.0
                if adv < min_adv:
                    continue

                # 単元コスト
                last_close = float(close.iloc[-1])
                cost_per_lot = last_close * lot_size
                if cost_per_lot > max_cost_per_lot:
                    continue

                # vol check — 日次ボラ
                ret = close.pct_change().dropna()
                if len(ret) < 10:
                    continue
                daily_vol = float(ret.std())
                if daily_vol < 0.003 or daily_vol > 0.08:
                    continue

                alive.append({
                    "symbol": sym,
                    "last_close": round(last_close, 1),
                    "adv_20d": round(adv, 0),
                    "daily_vol": round(daily_vol, 4),
                    "cost_per_lot": round(cost_per_lot, 0),
                    "data_days": len(close),
                })
            except Exception:
                continue

        # rate limit 対策
        if i + batch_size < total:
            time.sleep(1.0)

    return alive


def load_seed_universe() -> list[dict]:
    """db_update.py の TARGET_UNIVERSE を seed として読み込む。"""
    try:
        from db_update import TARGET_UNIVERSE
        return [
            {"symbol": sym, "name": name, "sector": sector, "source": "seed"}
            for sym, name, sector in TARGET_UNIVERSE
        ]
    except ImportError:
        return []


def build_universe(
    *,
    min_adv: float = 10_000_000,
    max_cost_per_lot: float = 500_000,
    lot_size: int = 100,
    use_jpx: bool = True,
    use_yfinance_scan: bool = True,
    output_path: str = "universe.json",
) -> dict:
    """Universe 構築メイン関数。"""
    print(f"[universe] 構築開始: min_adv={min_adv:,.0f}  max_cost={max_cost_per_lot:,.0f}")

    # Step 1: 候補コード取得
    candidates: dict[str, dict] = {}

    # 1a. seed (既存97銘柄) — 必ず含める
    seed = load_seed_universe()
    for item in seed:
        candidates[item["symbol"]] = item
    print(f"[universe] seed: {len(seed)} 銘柄")

    # 1b. JPX 公式リスト
    jpx_list = None
    if use_jpx:
        jpx_list = fetch_tse_from_jpx()
        if jpx_list:
            for item in jpx_list:
                sym = item["symbol"]
                if sym not in candidates:
                    candidates[sym] = item
                else:
                    # JPX データで name/sector を更新
                    if item.get("name"):
                        candidates[sym]["name"] = item["name"]
                    if item.get("sector"):
                        candidates[sym]["sector"] = item["sector"]
                    if item.get("market"):
                        candidates[sym]["market"] = item.get("market", "")
            print(f"[universe] JPX追加後: {len(candidates)} 銘柄")

    # 1c. コード範囲スキャン (JPX失敗時 or 追加)
    if use_yfinance_scan and jpx_list is None:
        print("[universe] JPX取得失敗 → yfinance コードスキャンモード")
        all_codes = generate_tse_codes()
        for code in all_codes:
            if code not in candidates:
                candidates[code] = {"symbol": code, "name": "", "sector": "", "source": "scan"}
        print(f"[universe] スキャン候補追加後: {len(candidates)} 銘柄")

    # Step 2: benchmark を分離（検証対象外だが最終出力には含める）
    benchmark_syms = {b[0] for b in BENCHMARK_TICKERS}
    verify_syms = [sym for sym in candidates if sym not in benchmark_syms]

    # Step 3: yfinance で一括データ検証 + 流動性フィルタ
    print(f"[universe] yfinance 検証開始: {len(verify_syms)} 銘柄")
    verified = verify_yfinance_batch(
        verify_syms,
        min_adv=min_adv,
        max_cost_per_lot=max_cost_per_lot,
        lot_size=lot_size,
    )
    verified_syms = {v["symbol"] for v in verified}
    verified_map = {v["symbol"]: v for v in verified}
    print(f"[universe] 検証通過: {len(verified)} 銘柄")

    # Step 4: 最終リスト構築
    output = []
    for item in verified:
        sym = item["symbol"]
        base = candidates.get(sym, {})
        output.append({
            "symbol": sym,
            "name": base.get("name", ""),
            "sector": base.get("sector", ""),
            "market": base.get("market", ""),
            "last_close": item.get("last_close"),
            "adv_20d": item.get("adv_20d"),
            "cost_per_lot": item.get("cost_per_lot"),
        })

    # benchmark 追加（検証なし）
    for sym, name, sector in BENCHMARK_TICKERS:
        if sym not in verified_syms:
            output.append({
                "symbol": sym,
                "name": name,
                "sector": sector,
                "market": "Benchmark",
            })

    # seed の中で検証落ちした銘柄を警告
    seed_syms = {s["symbol"] for s in seed} - benchmark_syms
    seed_dropped = seed_syms - verified_syms
    if seed_dropped:
        print(f"[universe] ⚠️  seed から落選: {sorted(seed_dropped)}")

    # ソート: sector → symbol
    output.sort(key=lambda x: (x.get("sector", ""), x["symbol"]))

    # メタデータ
    meta = {
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "built_date": date.today().isoformat(),
        "total_candidates_scanned": len(verify_syms),
        "verified_count": len(verified),
        "benchmark_count": len(BENCHMARK_TICKERS),
        "final_count": len(output),
        "seed_count": len(seed),
        "seed_dropped": sorted(seed_dropped),
        "jpx_available": jpx_list is not None,
        "filters": {
            "min_adv": min_adv,
            "max_cost_per_lot": max_cost_per_lot,
            "lot_size": lot_size,
        },
    }

    payload = {
        "_meta": meta,
        "universe": output,
    }

    Path(output_path).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[universe] 出力: {output_path}  ({len(output)} 銘柄)")

    return meta


def load_universe_from_file(path: str) -> list[tuple[str, str, str]]:
    """universe.json を db_update.py load_universe() 互換形式で読み込む。

    universe.json のフォーマット:
      {"_meta": {...}, "universe": [{"symbol": "...", "name": "...", "sector": "..."}, ...]}
    または従来形式:
      [{"symbol": "...", "name": "...", "sector": "..."}, ...]
    """
    raw = json.loads(Path(path).read_text(encoding="utf-8"))

    # 新フォーマット（_meta + universe キー）
    if isinstance(raw, dict) and "universe" in raw:
        items = raw["universe"]
    elif isinstance(raw, list):
        items = raw
    else:
        raise ValueError(f"Unknown universe.json format in {path}")

    result = []
    for it in items:
        if isinstance(it, dict):
            result.append((it["symbol"], it.get("name", ""), it.get("sector", "")))
        elif isinstance(it, (list, tuple)):
            result.append((it[0], it[1] if len(it) > 1 else "", it[2] if len(it) > 2 else ""))
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="TSE 動態 Universe 構築")
    ap.add_argument("--config", default=None, help="config.yaml パス")
    ap.add_argument("--output", default="universe.json", help="出力ファイル")
    ap.add_argument("--min-adv", type=float, default=None, help="最低 ADV (JPY)")
    ap.add_argument("--max-cost", type=float, default=None, help="最大単元コスト (JPY)")
    ap.add_argument("--no-jpx", action="store_true", help="JPX CSV を使用しない")
    ap.add_argument("--no-scan", action="store_true", help="yfinance コードスキャンを使用しない")
    args = ap.parse_args()

    # config からデフォルト読み込み
    min_adv = args.min_adv or 10_000_000
    max_cost = args.max_cost or 500_000

    if args.config:
        try:
            import yaml
            cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
            ucfg = cfg.get("universe", {}) or {}
            if not args.min_adv and ucfg.get("min_adv_universe"):
                min_adv = float(ucfg["min_adv_universe"])
            if not args.max_cost and ucfg.get("max_cost_per_lot_universe"):
                max_cost = float(ucfg["max_cost_per_lot_universe"])
        except Exception as e:
            print(f"[universe] config 読み込み失敗 (デフォルト使用): {e}", file=sys.stderr)

    meta = build_universe(
        min_adv=min_adv,
        max_cost_per_lot=max_cost,
        output_path=args.output,
        use_jpx=not args.no_jpx,
        use_yfinance_scan=not args.no_scan,
    )
    print(f"\n=== Universe Build Summary ===")
    print(f"  Scanned:  {meta['total_candidates_scanned']}")
    print(f"  Verified: {meta['verified_count']}")
    print(f"  Final:    {meta['final_count']} (incl {meta['benchmark_count']} benchmarks)")
    if meta["seed_dropped"]:
        print(f"  ⚠️  Seed dropped: {meta['seed_dropped']}")
