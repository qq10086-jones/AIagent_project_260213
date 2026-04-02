"""news_to_db.py — 免费新闻源 → SQLite news_feed / news_sentiment 桥接

将 Kabutan RSS / Google News RSS / GDELT 的日本股市情报写入 japan_market.db，
使 ss7_sqlite_news_overlay.py 的 news overlay 在不依赖付费 API 的情况下可正常工作。

数据源优先级（均免费）：
  1. Kabutan RSS (kabutan.jp) — 日文，含 TDnet 类别，~15min 延迟
  2. Google News RSS (news.google.com)  — 日文关键词，~5-30min 延迟
  3. GDELT DOC 2.0 API (gdeltproject.org) — 英文/多语，~15min 延迟

情感评分策略（无需 LLM / 付费 API）：
  - 基于 TDnet 事件类型的规则打分（上方修正 → +0.75，下方修正 → -0.75，等）
  - 如果本地 Ollama 可用（OLLAMA_BASE_URL 可访问），则用 LLM 对标题打 0/1/-1
  - 可通过 NEWS_USE_LLM=0 强制关闭 LLM

使用方法：
  python news_to_db.py --db japan_market.db [--lookback_hours 24] [--dry_run]

与 run_pipeline.py / daily_run.py 集成：
  config.yaml 中 news.enabled=true 时，pipeline 会在 screener 后调用本脚本。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sqlite3
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

try:
    import requests as _requests
    def _get(url: str, timeout: int = 8) -> Optional[bytes]:
        try:
            r = _requests.get(url, timeout=timeout, headers={"User-Agent": "Mozilla/5.0 (NexusBot/2.0)"})
            if r.status_code == 200:
                return r.content
        except Exception:
            pass
        return None
except ImportError:
    def _get(url: str, timeout: int = 8) -> Optional[bytes]:
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (NexusBot/2.0)"})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read()
        except Exception:
            pass
        return None


# ──────────────────────────────────────────────
# Ticker 名称反向查找（公司名 → symbol）
# ──────────────────────────────────────────────
def _build_name_map(db_path: str) -> dict[str, str]:
    """
    构建 {名称_小写: symbol} 映射，优先用 watchlist_alias.json，
    补充 tickers 表中的英文名。
    用于对不含代码的 Google News / GDELT 标题做 ticker 匹配。
    """
    name_map: dict[str, str] = {}

    # 1. watchlist_alias.json（含日文/英文别名）
    alias_path = Path(__file__).parent / "watchlist_alias.json"
    if alias_path.exists():
        try:
            aliases: dict[str, list[str]] = json.loads(alias_path.read_text(encoding="utf-8"))
            for symbol, names in aliases.items():
                for n in names:
                    key = n.lower().strip()
                    if key:
                        name_map[key] = symbol
        except Exception:
            pass

    # 2. tickers 表（英文名补充）
    if db_path and Path(db_path).exists():
        try:
            conn = sqlite3.connect(db_path)
            for symbol, name in conn.execute("SELECT symbol, name FROM tickers WHERE name IS NOT NULL").fetchall():
                key = str(name).lower().strip()
                if key and key not in name_map:
                    name_map[key] = symbol
            conn.close()
        except Exception:
            pass

    return name_map


def _match_ticker(title: str, name_map: dict[str, str]) -> Optional[str]:
    """
    从标题中提取 ticker：
    1. 直接匹配 4 位数字代码（任意括号格式 ‹4005›、<4005>、（4005）等）
    2. 对 name_map 做最长匹配（避免短名误触发）
    """
    # 先找代码
    m = re.search(r'[＜<（(\[【](\d{4})[＞>）)\]】]', title)
    if m:
        return m.group(1) + ".T"

    # 再做名称匹配（只匹配长度≥3 的名称，避免误触）
    title_l = title.lower()
    best: Optional[str] = None
    best_len = 2  # 最短匹配 3 个字符
    for name, symbol in name_map.items():
        if len(name) > best_len and name in title_l:
            best = symbol
            best_len = len(name)
    return best


# ──────────────────────────────────────────────
# TDnet 类别映射（用于规则打分）
# ──────────────────────────────────────────────
_CATEGORY_MAP = {
    "上方修正": ("earnings_revision_up",   +0.75),
    "下方修正": ("earnings_revision_down", -0.75),
    "増配":     ("dividend_increase",      +0.55),
    "特別配当": ("special_dividend",       +0.60),
    "自己株式": ("buyback",                +0.50),
    "自社株":   ("buyback",                +0.50),
    "業績修正": ("earnings_revision",       0.00),   # 方向不明，需 LLM 细化
    "決算短信": ("earnings",               +0.10),
    "M&A":      ("ma",                     +0.20),
    "業務提携": ("partnership",            +0.20),
    "不祥事":   ("scandal",               -0.80),
    "行政処分": ("regulatory",            -0.70),
    "減配":     ("dividend_cut",          -0.65),
}

_URGENCY_MAP = {
    "earnings_revision_up":   5.0,
    "earnings_revision_down": 5.0,
    "dividend_increase":      4.0,
    "special_dividend":       4.0,
    "buyback":                4.0,
    "earnings":               3.0,
    "ma":                     3.5,
    "partnership":            2.0,
    "dividend_cut":           4.5,
    "scandal":                5.0,
    "regulatory":             4.5,
    "earnings_revision":      3.0,
    "general":                1.0,
}


def _classify(title: str) -> tuple[str, float, float]:
    """返回 (category, sentiment_score, urgency)"""
    for kw, (cat, sent) in _CATEGORY_MAP.items():
        if kw in title:
            return cat, sent, _URGENCY_MAP.get(cat, 1.0)
    return "general", 0.0, 1.0


def _make_hash(title: str, url: str) -> str:
    return hashlib.sha1(f"{title}|{url}".encode("utf-8")).hexdigest()[:16]


def _parse_pubdate(pub: str) -> Optional[datetime]:
    if not pub:
        return None
    try:
        from email.utils import parsedate_to_datetime
        return parsedate_to_datetime(pub).replace(tzinfo=None)
    except Exception:
        pass
    for fmt in ("%Y%m%dT%H%M%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(pub[:19], fmt)
        except Exception:
            pass
    return None


# ──────────────────────────────────────────────
# Source 1: Kabutan RSS（最可靠的日本株式情报）
# ──────────────────────────────────────────────
def fetch_kabutan(lookback_hours: float = 24.0, name_map: dict | None = None) -> list[dict]:
    cutoff = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(hours=lookback_hours)
    urls = [
        "https://kabutan.jp/news/rss/?category=marketnews",
        "https://kabutan.jp/news/rss/?category=disclosure",
    ]
    items = []
    for rss_url in urls:
        data = _get(rss_url)
        if not data:
            continue
        try:
            root = ET.fromstring(data)
        except Exception:
            continue
        for item in root.iter("item"):
            title = (item.findtext("title") or "").strip()
            link  = (item.findtext("link")  or "").strip()
            pub   = (item.findtext("pubDate") or "").strip()
            if not title:
                continue
            pub_dt = _parse_pubdate(pub)
            if pub_dt and pub_dt < cutoff:
                continue
            code = _match_ticker(title + " " + link, name_map or {})
            cat, sent, urgency = _classify(title)
            items.append({
                "news_id":      _make_hash(title, link),
                "symbol":       code,
                "published_ts": pub_dt.isoformat() if pub_dt else datetime.now(timezone.utc).replace(tzinfo=None).isoformat(),
                "source":       "kabutan",
                "title":        title,
                "url":          link,
                "category":     cat,
                "sentiment":    sent,
                "urgency":      urgency,
            })
        if items:
            break   # Kabutan 成功则不继续
    return items


# ──────────────────────────────────────────────
# Source 2: Google News RSS（日语关键词）
# ──────────────────────────────────────────────
def fetch_google_news_jp(lookback_hours: float = 24.0, name_map: dict | None = None) -> list[dict]:
    cutoff = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(hours=lookback_hours)
    kw = "決算 OR 増配 OR 自社株買い OR 業績修正 OR 上方修正 OR 下方修正 OR 配当"
    url = f"https://news.google.com/rss/search?q={urllib.parse.quote(kw)}&hl=ja&gl=JP&ceid=JP:ja"
    data = _get(url)
    if not data:
        return []
    try:
        root = ET.fromstring(data)
    except Exception:
        return []
    items = []
    for item in root.iter("item"):
        title = (item.findtext("title") or "").strip()
        link  = (item.findtext("link")  or "").strip()
        pub   = (item.findtext("pubDate") or "").strip()
        if not title:
            continue
        pub_dt = _parse_pubdate(pub)
        if pub_dt and pub_dt < cutoff:
            continue
        code = _match_ticker(title, name_map or {})
        cat, sent, urgency = _classify(title)
        items.append({
            "news_id":      _make_hash(title, link),
            "symbol":       code,
            "published_ts": pub_dt.isoformat() if pub_dt else datetime.now(timezone.utc).replace(tzinfo=None).isoformat(),
            "source":       "google_news_jp",
            "title":        title,
            "url":          link,
            "category":     cat,
            "sentiment":    sent,
            "urgency":      urgency,
        })
    return items


# ──────────────────────────────────────────────
# Source 3: GDELT DOC 2.0 API（英文/多语，全球宏观）
# ──────────────────────────────────────────────
def fetch_gdelt(lookback_hours: float = 24.0, max_records: int = 30, name_map: dict | None = None) -> list[dict]:
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    start = now - timedelta(hours=lookback_hours)
    fmt = "%Y%m%d%H%M%S"
    query = "japan stock earnings OR nikkei OR TSE"
    url = (
        "https://api.gdeltproject.org/api/v2/doc/doc"
        f"?query={urllib.parse.quote(query)}"
        f"&mode=artlist&maxrecords={max_records}"
        f"&startdatetime={start.strftime(fmt)}&enddatetime={now.strftime(fmt)}"
        "&format=json&sort=DateDesc"
    )
    data = _get(url, timeout=12)
    if not data:
        return []
    try:
        obj = json.loads(data)
        arts = obj.get("articles") or []
    except Exception:
        return []
    items = []
    for art in arts:
        title = str(art.get("title") or "").strip()
        link  = str(art.get("url")   or "").strip()
        if not title or not link:
            continue
        pub_str = str(art.get("seendate") or "")
        pub_dt  = _parse_pubdate(pub_str)
        # 粗略情感：GDELT tone 字段为负数代表负面
        tone = float(art.get("tone") or 0.0)
        sentiment = max(-1.0, min(1.0, tone / 10.0))
        items.append({
            "news_id":      _make_hash(title, link),
            "symbol":       _match_ticker(title, name_map or {}),
            "published_ts": pub_dt.isoformat() if pub_dt else datetime.now(timezone.utc).replace(tzinfo=None).isoformat(),
            "source":       "gdelt",
            "title":        title,
            "url":          link,
            "category":     "general",
            "sentiment":    round(sentiment, 3),
            "urgency":      1.0,
        })
    return items


# ──────────────────────────────────────────────
# Source 4: Google News RSS — BOJ / 日银宏观情报
# ──────────────────────────────────────────────
def fetch_google_news_boj(lookback_hours: float = 24.0) -> list[dict]:
    """抓取日银/利率/货币政策相关新闻，impact_category = BOJ"""
    cutoff = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(hours=lookback_hours)
    kw = "日銀 OR BOJ OR 金利 OR 利上げ OR 利下げ OR 量的緩和 OR 金融政策"
    url = f"https://news.google.com/rss/search?q={urllib.parse.quote(kw)}&hl=ja&gl=JP&ceid=JP:ja"
    data = _get(url)
    if not data:
        return []
    try:
        root = ET.fromstring(data)
    except Exception:
        return []
    items = []
    for item in root.iter("item"):
        title = (item.findtext("title") or "").strip()
        link  = (item.findtext("link")  or "").strip()
        pub   = (item.findtext("pubDate") or "").strip()
        if not title:
            continue
        pub_dt = _parse_pubdate(pub)
        if pub_dt and pub_dt < cutoff:
            continue
        # BOJ 新闻：负面关键词检测
        neg_kw = ["利上げ", "引き締め", "インフレ", "円安リスク"]
        pos_kw = ["利下げ", "緩和", "景気支援"]
        sent = -0.3 if any(k in title for k in neg_kw) else \
               +0.3 if any(k in title for k in pos_kw) else 0.0
        items.append({
            "news_id":         _make_hash(title, link),
            "symbol":          None,
            "published_ts":    pub_dt.isoformat() if pub_dt else datetime.now(timezone.utc).replace(tzinfo=None).isoformat(),
            "source":          "google_news_boj",
            "title":           title,
            "url":             link,
            "category":        "boj_policy",
            "sentiment":       sent,
            "urgency":         4.0,
            "impact_category": "BOJ",
        })
    return items


# ──────────────────────────────────────────────
# Source 5: Google News RSS — 美日贸易/关税情报
# ──────────────────────────────────────────────
def fetch_google_news_trade(lookback_hours: float = 24.0) -> list[dict]:
    """抓取美日贸易/关税相关新闻，impact_category = TRADE"""
    cutoff = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(hours=lookback_hours)
    kw = "関税 OR 貿易摩擦 OR 日米貿易 OR トランプ 関税 OR 輸出規制 OR 半導体規制"
    url = f"https://news.google.com/rss/search?q={urllib.parse.quote(kw)}&hl=ja&gl=JP&ceid=JP:ja"
    data = _get(url)
    if not data:
        return []
    try:
        root = ET.fromstring(data)
    except Exception:
        return []
    items = []
    for item in root.iter("item"):
        title = (item.findtext("title") or "").strip()
        link  = (item.findtext("link")  or "").strip()
        pub   = (item.findtext("pubDate") or "").strip()
        if not title:
            continue
        pub_dt = _parse_pubdate(pub)
        if pub_dt and pub_dt < cutoff:
            continue
        neg_kw = ["関税", "規制", "制裁", "摩擦", "禁輸"]
        sent = -0.4 if any(k in title for k in neg_kw) else 0.0
        items.append({
            "news_id":         _make_hash(title, link),
            "symbol":          None,
            "published_ts":    pub_dt.isoformat() if pub_dt else datetime.now(timezone.utc).replace(tzinfo=None).isoformat(),
            "source":          "google_news_trade",
            "title":           title,
            "url":             link,
            "category":        "trade_policy",
            "sentiment":       sent,
            "urgency":         4.5,
            "impact_category": "TRADE",
        })
    return items


# ──────────────────────────────────────────────
# 可选：Ollama LLM 二次评分（本地免费）
# ──────────────────────────────────────────────
def _llm_score_titles(titles: list[str]) -> list[float]:
    """
    用本地 Ollama 对标题打分：+1 利多 / 0 中性 / -1 利空。
    如果 Ollama 不可用直接返回空列表（不影响主流程）。
    """
    base = os.getenv("OLLAMA_BASE_URL", "http://host.docker.internal:11434")
    model = os.getenv("QUANT_LLM_MODEL", "deepseek-r1:1.5b")
    prompt = (
        "以下是日本股票相关新闻标题列表，每行一条。"
        "请对每条给出情感得分：+1（利多）、0（中性）、-1（利空）。"
        "只输出一个 JSON 数组，例如 [1, 0, -1, ...]，不要其他文字。\n\n"
        + "\n".join(f"{i+1}. {t}" for i, t in enumerate(titles[:20]))
    )
    payload = json.dumps({"model": model, "prompt": prompt, "stream": False})
    try:
        req = urllib.request.Request(
            f"{base}/api/generate",
            data=payload.encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=20) as resp:
            body = json.loads(resp.read())
        text = body.get("response", "")
        # 提取 JSON 数组
        m = re.search(r'\[[\s\-\d,]+\]', text)
        if m:
            scores = json.loads(m.group())
            return [float(s) for s in scores]
    except Exception:
        pass
    return []


# ──────────────────────────────────────────────
# 写入 SQLite
# ──────────────────────────────────────────────
def write_to_db(db_path: str, items: list[dict], dry_run: bool = False, model_version: str = "rules_v1") -> int:
    """
    将 news items 写入 news_feed + news_sentiment 表（去重）。
    返回实际写入行数。
    """
    if not items:
        return 0

    ingested_ts = datetime.now(timezone.utc).replace(tzinfo=None).isoformat()

    # 可选：对有 symbol 的标题做 LLM 二次评分
    use_llm = os.getenv("NEWS_USE_LLM", "1") not in ("0", "false", "no")
    sym_items = [it for it in items if it.get("symbol")]
    if use_llm and sym_items:
        titles = [it["title"] for it in sym_items]
        llm_scores = _llm_score_titles(titles)
        if llm_scores and len(llm_scores) >= len(sym_items):
            for it, sc in zip(sym_items, llm_scores):
                # 融合规则分与 LLM 分（各占 50%）
                it["sentiment"] = round((it["sentiment"] + sc) / 2.0, 3)
            model_version = "hybrid_rules_llm_v1"

    if dry_run:
        for it in items:
            sym = it.get("symbol") or "?"
            print(f"[dry_run] {it['source']:20s} {sym:8s} sent={it['sentiment']:+.2f}  {it['title'][:60]}")
        return len(items)

    conn = sqlite3.connect(db_path)
    try:
        conn.execute("PRAGMA journal_mode=WAL;")
        # 确保表存在（直接 CREATE IF NOT EXISTS，不依赖 trade_schema import）
        conn.execute("""
            CREATE TABLE IF NOT EXISTS news_feed (
                news_id TEXT PRIMARY KEY,
                symbol TEXT,
                published_ts TEXT NOT NULL,
                source TEXT NOT NULL,
                title TEXT NOT NULL,
                content_summary TEXT,
                url TEXT,
                ingested_ts TEXT NOT NULL,
                event_cluster_id TEXT,
                raw_hash TEXT
            )""")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS news_sentiment (
                news_id TEXT,
                model_version TEXT,
                sentiment_score REAL,
                urgency REAL,
                expected_impact_days REAL,
                reason TEXT,
                scored_ts TEXT,
                PRIMARY KEY (news_id, model_version)
            )""")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_news_feed_sym_ts ON news_feed(symbol, published_ts)")
        # v2 标准化表（供 quant_briefing.py _cross_validate_with_news 使用）
        conn.execute("""
            CREATE TABLE IF NOT EXISTS news_items (
                news_id          TEXT PRIMARY KEY,
                related_tickers  TEXT DEFAULT '[]',
                impact_category  TEXT NOT NULL DEFAULT 'MARKET_WIDE',
                sentiment_score  REAL NOT NULL DEFAULT 0.0,
                summary_cn       TEXT DEFAULT '',
                published_at     TEXT NOT NULL,
                source           TEXT NOT NULL,
                urgency          REAL DEFAULT 1.0
            )""")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_news_items_cat_ts ON news_items(impact_category, published_at)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_news_items_tickers ON news_items(related_tickers)")

        written = 0
        for it in items:
            nid = it["news_id"]
            # news_feed（去重）
            conn.execute(
                "INSERT OR IGNORE INTO news_feed "
                "(news_id, symbol, published_ts, source, title, url, ingested_ts) "
                "VALUES (?,?,?,?,?,?,?)",
                (nid, it.get("symbol"), it["published_ts"], it["source"],
                 it["title"], it.get("url", ""), ingested_ts)
            )
            # news_sentiment（去重）
            conn.execute(
                "INSERT OR IGNORE INTO news_sentiment "
                "(news_id, model_version, sentiment_score, urgency, expected_impact_days, reason, scored_ts) "
                "VALUES (?,?,?,?,?,?,?)",
                (nid, model_version, it["sentiment"], it["urgency"],
                 3.0,   # 默认影响 3 个交易日
                 it["category"], ingested_ts)
            )
            # news_items（v2 标准化表，供 quant_briefing.py 交叉验证）
            sym = it.get("symbol")
            # impact_category 推导逻辑
            impact_cat = it.get("impact_category")  # 已有字段（BOJ/TRADE 源直接设置）
            if impact_cat is None:
                src = it.get("source", "")
                if sym:
                    impact_cat = "COMPANY"
                elif src == "gdelt":
                    impact_cat = "MARKET_WIDE"
                else:
                    impact_cat = "SECTOR"
            related_tickers = f'["{sym}"]' if sym else "[]"
            # summary_cn = 标题截断（LLM摘要可选，此处用原标题作为占位）
            summary_cn = (it.get("title") or "")[:50]
            conn.execute(
                "INSERT OR IGNORE INTO news_items "
                "(news_id, related_tickers, impact_category, sentiment_score, summary_cn, published_at, source, urgency) "
                "VALUES (?,?,?,?,?,?,?,?)",
                (nid, related_tickers, impact_cat, it["sentiment"],
                 summary_cn, it["published_ts"], it.get("source", ""), it["urgency"])
            )
            written += 1
        conn.commit()
    finally:
        conn.close()

    return written


# ──────────────────────────────────────────────
# 主入口
# ──────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="写入日本股市新闻情感数据到 SQLite")
    ap.add_argument("--db",             default="japan_market.db")
    ap.add_argument("--lookback_hours", type=float, default=24.0,
                    help="抓取过去多少小时的新闻（默认 24h）")
    ap.add_argument("--dry_run",        action="store_true",
                    help="只打印，不写入数据库")
    ap.add_argument("--sources",        default="kabutan,google,boj,trade,gdelt",
                    help="逗号分隔的数据源: kabutan,google,boj,trade,gdelt")
    args = ap.parse_args()

    all_items: list[dict] = []
    seen_ids: set[str] = set()

    # 构建公司名 → ticker 映射（利用 watchlist_alias.json + tickers 表）
    name_map = _build_name_map(args.db)
    print(f"[name_map]    加载 {len(name_map)} 个别名")

    sources = [s.strip().lower() for s in args.sources.split(",")]
    for src in sources:
        if src == "kabutan":
            fetched = fetch_kabutan(args.lookback_hours, name_map=name_map)
            print(f"[kabutan]     抓取 {len(fetched)} 条")
        elif src == "google":
            fetched = fetch_google_news_jp(args.lookback_hours, name_map=name_map)
            print(f"[google_jp]   抓取 {len(fetched)} 条")
        elif src == "boj":
            fetched = fetch_google_news_boj(args.lookback_hours)
            print(f"[google_boj]  抓取 {len(fetched)} 条")
        elif src == "trade":
            fetched = fetch_google_news_trade(args.lookback_hours)
            print(f"[google_trade] 抓取 {len(fetched)} 条")
        elif src == "gdelt":
            fetched = fetch_gdelt(args.lookback_hours, name_map=name_map)
            print(f"[gdelt]       抓取 {len(fetched)} 条")
        else:
            print(f"[warn] 未知数据源: {src}，跳过")
            continue

        for it in fetched:
            if it["news_id"] not in seen_ids:
                seen_ids.add(it["news_id"])
                all_items.append(it)

    sym_count = sum(1 for it in all_items if it.get("symbol"))
    print(f"[total] 合并后 {len(all_items)} 条（含 ticker 的 {sym_count} 条）")

    n = write_to_db(args.db, all_items, dry_run=args.dry_run)
    if not args.dry_run:
        print(f"[db] 写入 {n} 条到 {args.db}")


if __name__ == "__main__":
    main()
