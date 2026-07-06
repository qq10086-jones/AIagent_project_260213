"""HTR-native news -> theme / macro classifier.

WHY: the news pipeline ingests broad news (Google News JP) but ~89% of items
carry no `related_tickers`, so they are dropped from the ticker-centric view.
A large share of that 89% is exactly the macro / policy / sector news that
should DRIVE hot-theme rotation (e.g. "経産省 半導体補助金" -> semi theme up).
This module routes that previously-lost news into theme-level and macro-level
signals.

DESIGN (P10-26, first slice):
- Deterministic keyword taxonomy — NO LLM, NO GPU. Transparent + auditable
  (Rule 8.3 prefers transparent mappings over black-box scores; the
  no-background-LLM-batch constraint means the bulk classifier must not need
  the shared GPU). An optional LLM refinement layer is a separate,
  user-triggered step.
- ADR-0005: strictly read-only on Project_optimized's `news_feed` (mode=ro);
  the overlay is HTR-native output under `reports/news_themes/`.
- Rule 8.2 PIT: every item keeps its `published_ts`; the overlay is built for a
  point-in-time window and never looks past it.
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any


class NewsThemeClassifierError(RuntimeError):
    """Raised when news_feed cannot be safely read."""


# Eight themes mirror the dashboard taxonomy (semi / ai / auto / bank / defense /
# energy / optical / memory). Keywords matched case-insensitively as substrings
# (JP + CN + EN). optical/memory added 2026-06-15 (Event Desk E1) so the engine
# can SEE the AI-interconnect (光模块) and storage (HBM/DRAM/NAND) narratives that
# were previously invisible. Compound keywords only — bare "光"/"メモリ" are too broad.
THEME_TAXONOMY: dict[str, tuple[str, ...]] = {
    "semi": ("半導体", "半导体", "セミコン", "semiconductor", "chip", "tsmc", "asml",
             "nvidia", "エヌビディア", "ファウンドリ", "ウエハ", "露光", "euv", "sox",
             "東京エレクトロン", "レーザーテック", "アドバンテスト"),
    "ai": ("ai", "人工知能", "生成ai", "データセンター", "gpu", "arm", "llm", "chatgpt",
           "ソフトバンクグループ", "クラウド"),
    "auto": ("自動車", "汽车", " ev ", "トヨタ", "ホンダ", "日産", "自動車部品",
             "電気自動車", "ハイブリッド"),
    "bank": ("銀行", "メガバンク", "三菱ufj", "三井住友", "みずほ", "金融機関", "保険",
             "証券", "信託"),
    "defense": ("防衛", "軍事", "防衛費", "三菱重工", "川崎重工", "防衛装備"),
    "energy": ("エネルギー", "総合商社", "石油", "原油", "三菱商事", "伊藤忠", "lng",
               "天然ガス", "電力", "再生可能エネルギー"),
    "optical": ("光モジュール", "光通信", "光トランシーバ", "シリコンフォトニクス",
                "光半導体", "光ファイバ", "光部品", "光インターコネクト", "co-packaged",
                "cpo", "光模块", "硅光", "光器件", "光通讯", "フジクラ", "古河電工",
                "コヒレント", "lumentum", "transceiver"),
    "memory": ("dram", "nand", "hbm", "半導体メモリ", "フラッシュメモリ", "メモリ半導体",
               "キオクシア", "kioxia", "ストレージ", "ssd", "存储", "内存", "闪存",
               "memory chip", "広帯域メモリ"),
}

# Macro categories feed market temperature + cross-cut every theme.
MACRO_TAXONOMY: dict[str, tuple[str, ...]] = {
    "monetary": ("日銀", "日本銀行", "利上げ", "利下げ", "金利", "金融政策", "boj",
                 "金融緩和", "植田", "マイナス金利"),
    "fiscal": ("経産省", "meti", "補助金", "政府", "予算", "財務省", "規制緩和", "減税",
               "経済対策"),
    "fx": ("円安", "円高", "ドル円", "為替", "usd/jpy", "為替介入"),
    "trade": ("関税", "貿易摩擦", "輸出規制", "米中", "通商"),
    "overseas": ("frb", "fomc", "米国株", "ダウ平均", "ナスダック", "s&p500",
                 "雇用統計", "cpi", "fed"),
    # geopolitics added 2026-06-15 (Event Desk E1) — ceasefire / Middle East /
    # sanctions move oil + defense risk premium, which the macro layer was blind to.
    "geopolitics": ("地政学", "地缘", "中東", "中东", "イラン", "伊朗", "停戦", "停战",
                    "有事", "制裁", "紛争", "ウクライナ", "台湾有事", "ホルムズ", "ceasefire"),
}


def _match(text: str, keywords: tuple[str, ...]) -> list[str]:
    t = (text or "").lower()
    return [k for k in keywords if k.lower() in t]


def classify_news(title: str, summary: str = "") -> dict[str, list[str]]:
    """Classify one news item into theme ids + macro category ids by keyword.

    Returns {"themes": [...], "macro": [...]}. Either may be empty (truly
    unclassifiable / pure single-stock noise).
    """
    text = f"{title or ''} {summary or ''}"
    themes = [tid for tid, kws in THEME_TAXONOMY.items() if _match(text, kws)]
    macro = [mid for mid, kws in MACRO_TAXONOMY.items() if _match(text, kws)]
    return {"themes": themes, "macro": macro}


def _validate_db(db_path: str | Path) -> Path:
    src = Path(db_path)
    if not src.exists():
        raise NewsThemeClassifierError(f"japan_market.db not found: {src}")
    return src


def read_recent_news(
    db_path: str | Path,
    *,
    asof: str | None = None,
    hours: int = 48,
    limit: int = 2000,
) -> tuple[dict[str, Any], ...]:
    """Read recent news_feed rows (read-only). Window = [asof-hours, asof]; if
    asof is None, anchor on the newest row so it works on a static snapshot DB."""
    src = _validate_db(db_path)
    conn = sqlite3.connect(f"file:{src}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        if not conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='news_feed'"
        ).fetchone():
            raise NewsThemeClassifierError("missing required table: news_feed")
        anchor = asof or (conn.execute(
            "SELECT MAX(published_ts) FROM news_feed"
        ).fetchone()[0])
        if not anchor:
            return ()
        cutoff = (datetime.fromisoformat(str(anchor)[:19]) - timedelta(hours=hours)).isoformat()
        rows = conn.execute(
            """
            SELECT news_id, symbol, published_ts, source, title, content_summary, url
            FROM news_feed
            WHERE published_ts >= ? AND published_ts <= ?
            ORDER BY published_ts DESC
            LIMIT ?
            """,
            (cutoff, str(anchor), int(limit)),
        ).fetchall()
    finally:
        conn.close()
    return tuple(dict(r) for r in rows)


def build_theme_news_overlay(
    db_path: str | Path,
    *,
    asof: str | None = None,
    hours: int = 48,
    base_dir: str | Path | None = None,
    max_recent: int = 8,
) -> dict[str, Any]:
    """Classify recent news into a theme/macro overlay. Optionally writes the
    HTR-native artifact `reports/news_themes/{date}.json` when base_dir is given.

    The overlay surfaces, per theme and macro category: how much fresh news
    points at it (count) and the most-recent headlines (with PIT timestamps),
    so the previously-dropped macro/sector news can drive theme heat.
    """
    rows = read_recent_news(db_path, asof=asof, hours=hours)
    themes: dict[str, dict[str, Any]] = {t: {"news_count": 0, "recent": []} for t in THEME_TAXONOMY}
    macro: dict[str, dict[str, Any]] = {m: {"news_count": 0, "recent": []} for m in MACRO_TAXONOMY}
    classified = 0
    for r in rows:
        c = classify_news(r.get("title", ""), r.get("content_summary", ""))
        if c["themes"] or c["macro"]:
            classified += 1
        item = {
            "ts": r.get("published_ts"),
            "title": r.get("title"),
            "source": r.get("source"),
            "url": r.get("url"),
            "symbol": r.get("symbol"),
            "themes": c["themes"],
            "macro": c["macro"],
        }
        for t in c["themes"]:
            themes[t]["news_count"] += 1
            if len(themes[t]["recent"]) < max_recent:
                themes[t]["recent"].append(item)
        for m in c["macro"]:
            macro[m]["news_count"] += 1
            if len(macro[m]["recent"]) < max_recent:
                macro[m]["recent"].append(item)

    overlay = {
        "asof": asof or (rows[0]["published_ts"] if rows else None),
        "window_hours": hours,
        "total_news": len(rows),
        "classified_news": classified,
        "unclassified_news": len(rows) - classified,
        "themes": themes,
        "macro": macro,
        "method": "deterministic_keyword_taxonomy_v1",
        "note": "Rule 8.2 PIT window; Rule 8.3 transparent mapping (no LLM); ADR-0005 read-only source.",
    }

    if base_dir is not None:
        out_dir = Path(base_dir) / "reports" / "news_themes"
        out_dir.mkdir(parents=True, exist_ok=True)
        day = (overlay["asof"] or "unknown")[:10]
        (out_dir / f"{day}.json").write_text(
            json.dumps(overlay, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    return overlay
