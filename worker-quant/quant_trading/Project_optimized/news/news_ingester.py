"""news_ingester.py

Standalone news ingestion script for the quant trading project.

Fetches news from Google RSS for each tracked symbol and persists raw
articles to the `news_feed` table in SQLite.  Deduplication is done via
SHA-256 hash of (title + url) so the same article is never inserted twice.

Governance compliance (news/design/NEWS_GOVERNANCE_v1.md):
  Rule 1.1  : Every row written includes both published_ts and ingested_ts.
  Rule 2.1  : Per-symbol failures are caught; run continues for remaining symbols.
  Rule 2.2  : Degradation alarm when daily count falls below dynamic threshold.
  Rule 3.1  : User-Agent identifies this as QuantNewsBot/1.0.
  Rule 3.2  : ≤ 1 req/sec between symbol requests; max 3 retries with backoff.
  Rule 3.3  : robots.txt checked before each domain; forbidden paths are skipped.

Data sources:
  google_news_rss  Google News RSS (EN + JA for .T symbols)
  tdnet            TDnet official disclosure RDF feed (Japan listed companies)
  jquants          J-Quants API financial announcements (requires JQUANTS_REFRESH_TOKEN)

Usage
-----
  # from Project_optimized/ root:
  python news/news_ingester.py --db japan_market.db
  python news/news_ingester.py --db japan_market.db --symbols 9432.T,9433.T
  python news/news_ingester.py --db japan_market.db --max-per-symbol 10
"""
from __future__ import annotations

import argparse
import hashlib
import os
import re
import sqlite3
import sys
import time
import uuid
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import List, Optional

import urllib.parse
import urllib.robotparser

import requests

# Allow running from any working directory: ensure Project_optimized/ is on sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))
from trade_schema import connect, ensure_news_tables


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_USER_AGENT = "Mozilla/5.0 (compatible; QuantNewsBot/1.0)"
_REQUEST_TIMEOUT = 20
_RETRY_DELAYS = (2, 4, 8)          # seconds between retries (exponential backoff)
_SYMBOL_INTERVAL = 1.0             # seconds between symbol requests (Governance Rule 3.2)
_DEGRADATION_LOOKBACK_DAYS = 90    # days used to compute P5 threshold
_COLD_START_RATIO = 0.30           # fallback ratio when < 90 days history


# ---------------------------------------------------------------------------
# robots.txt compliance (Governance Rule 3.3)
# ---------------------------------------------------------------------------

class _RobotsCache:
    """Per-run in-memory cache of parsed robots.txt files, keyed by domain."""

    def __init__(self) -> None:
        self._cache: dict[str, urllib.robotparser.RobotFileParser] = {}

    def is_allowed(self, url: str) -> bool:
        """Return True if _USER_AGENT is permitted to fetch *url* per robots.txt."""
        parsed = urllib.parse.urlparse(url)
        domain = f"{parsed.scheme}://{parsed.netloc}"
        if domain not in self._cache:
            rp = urllib.robotparser.RobotFileParser()
            robots_url = f"{domain}/robots.txt"
            rp.set_url(robots_url)
            try:
                rp.read()
            except Exception:
                # If robots.txt is unreachable, assume allowed (fail-open for governance doc)
                self._cache[domain] = None  # type: ignore[assignment]
                return True
            self._cache[domain] = rp
        rp = self._cache[domain]
        if rp is None:
            return True
        return rp.can_fetch(_USER_AGENT, url)


_robots = _RobotsCache()


def _is_allowed(url: str) -> bool:
    """Check robots.txt; log and return False if disallowed (Governance Rule 3.3)."""
    if not _robots.is_allowed(url):
        print(f"    [SKIP_ROBOTS_TXT] {url}")
        return False
    return True


# ---------------------------------------------------------------------------
# HTTP helpers — with exponential backoff retry
# ---------------------------------------------------------------------------

def _get(url: str, params: dict | None = None) -> Optional[requests.Response]:
    """GET with up to 3 retries on failure (2s / 4s / 8s backoff).

    Returns None if all attempts fail (Governance Rule 2.1: never raises).
    """
    last_exc: Optional[Exception] = None
    for attempt in range(len(_RETRY_DELAYS) + 1):
        if attempt > 0:
            wait = _RETRY_DELAYS[attempt - 1]
            print(f"    [retry {attempt}/{len(_RETRY_DELAYS)}] waiting {wait}s...")
            time.sleep(wait)
        try:
            resp = requests.get(
                url,
                params=params,
                timeout=_REQUEST_TIMEOUT,
                headers={"User-Agent": _USER_AGENT},
            )
            if resp.status_code == 200:
                return resp
            # 429 or 5xx — treat as retriable
            last_exc = Exception(f"HTTP {resp.status_code}")
        except Exception as exc:
            last_exc = exc

    print(f"    [warn] all retries exhausted: {last_exc}")
    return None


# ---------------------------------------------------------------------------
# Timestamp parsing
# ---------------------------------------------------------------------------

def _parse_pub_ts(raw: str) -> Optional[str]:
    """Return ISO-8601 UTC string from RFC-2822 or GDELT compact format."""
    if not raw:
        return None
    raw = raw.strip()
    try:
        dt = parsedate_to_datetime(raw)
        return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    except Exception:
        pass
    m = re.match(r"^(\d{4})(\d{2})(\d{2})T(\d{2})(\d{2})(\d{2})Z?$", raw)
    if m:
        return (
            f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
            f"T{m.group(4)}:{m.group(5)}:{m.group(6)}Z"
        )
    return None


def _now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _today_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


# ---------------------------------------------------------------------------
# News fetching
# ---------------------------------------------------------------------------

def _fetch_google_rss(
    symbol: str,
    company_name: Optional[str] = None,
    max_items: int = 10,
    lang: str = "en",
) -> List[dict]:
    """Fetch news items from Google News RSS for a single symbol."""
    terms = [symbol]
    if company_name:
        terms.append(company_name)
    if lang != "ja":
        terms.append("stock")
    q = " ".join(t for t in terms if t)

    if lang == "ja":
        params = {"q": q, "hl": "ja", "gl": "JP", "ceid": "JP:ja"}
    elif lang == "zh":
        params = {"q": q, "hl": "zh-CN", "gl": "CN", "ceid": "CN:zh-Hans"}
    else:
        params = {"q": q, "hl": "en-US", "gl": "US", "ceid": "US:en"}

    target_url = "https://news.google.com/rss/search"
    if not _is_allowed(target_url):
        return []
    resp = _get(target_url, params=params)
    if resp is None:
        return []

    try:
        root = ET.fromstring(resp.content)
    except ET.ParseError:
        return []

    items = []
    for el in root.findall(".//item")[:max_items]:
        title = (el.findtext("title") or "").strip()
        link = (el.findtext("link") or "").strip()
        publisher = (el.findtext("source") or "").strip()
        pub_date = (el.findtext("pubDate") or "").strip()
        if not title or not link:
            continue
        items.append({
            "title": title,
            "url": link,
            "publisher": publisher,
            "published_ts": _parse_pub_ts(pub_date),
            "source": "google_news_rss",
        })
    return items


# ---------------------------------------------------------------------------
# TDnet RSS (Japan official disclosure feed)
# ---------------------------------------------------------------------------

_TDNET_RSS_URL = "https://www.release.tdnet.info/inbs/I_rss.rdf"
# RDF namespaces used by TDnet
_NS = {
    "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
    "dc":  "http://purl.org/dc/elements/1.1/",
}

# Module-level cache: fetch TDnet feed once per run, shared across symbols
_tdnet_feed_cache: Optional[list] = None


def _load_tdnet_feed() -> list:
    """Fetch and parse TDnet RDF RSS once per process; cache result."""
    global _tdnet_feed_cache
    if _tdnet_feed_cache is not None:
        return _tdnet_feed_cache

    if not _is_allowed(_TDNET_RSS_URL):
        _tdnet_feed_cache = []
        return []

    resp = _get(_TDNET_RSS_URL)
    if resp is None:
        _tdnet_feed_cache = []
        return []

    try:
        root = ET.fromstring(resp.content)
    except ET.ParseError:
        _tdnet_feed_cache = []
        return []

    # Try namespaced RDF items first, fall back to bare <item> tags
    _RDF_NS  = "http://www.w3.org/1999/02/22-rdf-syntax-ns#"
    _DC_NS   = "http://purl.org/dc/elements/1.1/"
    els = root.findall(f"{{{_RDF_NS}}}item")
    if not els:
        els = root.findall(f".//{{{_RDF_NS}}}item")
    if not els:
        els = root.findall(".//item")

    if not els:
        print(f"    [warn] TDnet: no <item> elements found in feed (root tag: {root.tag})")

    items = []
    for el in els:
        # Title: try bare, then dc: namespace
        title = (
            el.findtext("title")
            or el.findtext(f"{{{_DC_NS}}}title")
            or ""
        ).strip()
        # Link: try rdf:about attribute, then bare <link>
        link = (
            el.get(f"{{{_RDF_NS}}}about")
            or el.findtext("link")
            or ""
        ).strip()
        # Date: dc:date preferred, then pubDate
        date = (
            el.findtext(f"{{{_DC_NS}}}date")
            or el.findtext("pubDate")
            or ""
        ).strip()
        # Description: dc: preferred, then bare
        desc = (
            el.findtext(f"{{{_DC_NS}}}description")
            or el.findtext("description")
            or ""
        ).strip()

        if not title or not link:
            continue
        items.append({
            "title": title,
            "url": link,
            "publisher": "TDnet",
            "published_ts": _parse_pub_ts(date),
            "source": "tdnet",
            "description": desc,
        })

    _tdnet_feed_cache = items
    return items


def _fetch_tdnet_rss(symbol: str, max_items: int = 5) -> List[dict]:
    """Return TDnet disclosures matching *symbol* (4-digit TSE code)."""
    if not symbol.endswith(".T"):
        return []

    code = symbol[:-2]  # "9432.T" → "9432"
    # Word-boundary regex prevents "9432" matching "94321" or "/path94326/"
    code_pattern = re.compile(r"(?<!\d)" + re.escape(code) + r"(?!\d)")
    all_items = _load_tdnet_feed()

    matched = []
    for item in all_items:
        haystack = f"{item['title']} {item['url']} {item.get('description', '')}"
        if code_pattern.search(haystack):
            matched.append({k: v for k, v in item.items() if k != "description"})
            if len(matched) >= max_items:
                break
    return matched


# ---------------------------------------------------------------------------
# J-Quants API (optional — requires JQUANTS_REFRESH_TOKEN env var)
# ---------------------------------------------------------------------------

_JQUANTS_BASE = "https://api.jquants.com/v1"
_jquants_id_token: Optional[str] = None


def _jquants_get_id_token() -> Optional[str]:
    """Exchange refresh token for id_token (session-scoped)."""
    global _jquants_id_token
    if _jquants_id_token:
        return _jquants_id_token

    refresh_token = os.environ.get("JQUANTS_REFRESH_TOKEN")
    if not refresh_token:
        return None

    resp = _get(
        f"{_JQUANTS_BASE}/token/auth_refresh",
        params={"refreshtoken": refresh_token},
    )
    if resp is None:
        return None
    try:
        _jquants_id_token = resp.json().get("idToken")
    except Exception:
        return None
    return _jquants_id_token


def _fetch_jquants_announcements(symbol: str, max_items: int = 5) -> List[dict]:
    """Fetch financial announcements from J-Quants API for a .T symbol.

    Requires JQUANTS_REFRESH_TOKEN env var.  Silently returns [] if
    the token is absent or the request fails.
    """
    if not symbol.endswith(".T"):
        return []

    id_token = _jquants_get_id_token()
    if not id_token:
        return []

    code = symbol[:-2]  # "9432.T" → "9432"
    resp = _get(
        f"{_JQUANTS_BASE}/fins/announcement",
        params={"code": code},
    )
    if resp is None:
        return []

    try:
        data = resp.json()
    except Exception:
        return []

    items = []
    for entry in (data.get("announcement") or [])[:max_items]:
        title = entry.get("CompanyName", code) + " - " + entry.get("DocumentType", "Announcement")
        date_str = entry.get("Date", "")
        url = entry.get("URL") or f"{_JQUANTS_BASE}/fins/announcement?code={code}"
        items.append({
            "title": title,
            "url": url,
            "publisher": "J-Quants",
            "published_ts": _parse_pub_ts(date_str),
            "source": "jquants",
        })
    return items


def fetch_news_for_symbol(
    symbol: str,
    company_name: Optional[str] = None,
    max_per_symbol: int = 10,
) -> List[dict]:
    """Aggregate and deduplicate news for one symbol."""
    articles: List[dict] = []
    articles.extend(_fetch_google_rss(symbol, company_name, max_items=max_per_symbol, lang="en"))
    if symbol.endswith(".T"):
        articles.extend(
            _fetch_google_rss(symbol, company_name, max_items=max_per_symbol // 2, lang="ja")
        )
        articles.extend(_fetch_tdnet_rss(symbol, max_items=max_per_symbol // 2))
        articles.extend(_fetch_jquants_announcements(symbol, max_items=max_per_symbol // 4))

    seen: set[tuple[str, str]] = set()
    unique = []
    for a in articles:
        key = (a["title"].lower()[:120], a["url"].lower())
        if key not in seen:
            seen.add(key)
            unique.append(a)
    return unique


# ---------------------------------------------------------------------------
# SHA-256 raw hash
# ---------------------------------------------------------------------------

def _raw_hash(title: str, url: str) -> str:
    blob = f"{title.strip()}\n{url.strip()}".encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


# ---------------------------------------------------------------------------
# Symbol list
# ---------------------------------------------------------------------------

def get_tracked_symbols(conn: sqlite3.Connection) -> List[str]:
    """Return all symbols present in daily_prices."""
    try:
        rows = conn.execute(
            "SELECT DISTINCT symbol FROM daily_prices ORDER BY symbol"
        ).fetchall()
        return [r[0] for r in rows]
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def ingest_articles(
    conn: sqlite3.Connection,
    symbol: str,
    articles: List[dict],
    ingested_ts: str,
) -> tuple[int, int]:
    """Insert articles into news_feed; skip exact duplicates via raw_hash.

    Returns (inserted, skipped).
    """
    inserted = skipped = 0
    for art in articles:
        title = art.get("title", "").strip()
        url = art.get("url", "").strip()
        if not title:
            skipped += 1
            continue

        rh = _raw_hash(title, url)
        if conn.execute("SELECT 1 FROM news_feed WHERE raw_hash = ?", (rh,)).fetchone():
            skipped += 1
            continue

        pub_ts = art.get("published_ts") or ingested_ts
        try:
            conn.execute(
                """
                INSERT INTO news_feed
                    (news_id, symbol, published_ts, source, title,
                     content_summary, url, ingested_ts, raw_hash)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(uuid.uuid4()),
                    symbol,
                    pub_ts,
                    art.get("source", "google_news_rss"),
                    title,
                    art.get("publisher", ""),
                    url,
                    ingested_ts,
                    rh,
                ),
            )
            inserted += 1
        except sqlite3.IntegrityError:
            skipped += 1

    conn.commit()
    return inserted, skipped


# ---------------------------------------------------------------------------
# Degradation check (Governance Rule 2.2)
# ---------------------------------------------------------------------------

def check_degradation(conn: sqlite3.Connection, today_count: int) -> Optional[str]:
    """Return a DATA_DEGRADED warning string if today's count is below threshold.

    Threshold:
      >= 90 days history : P5 percentile of daily counts
      <  90 days history : 30-day average * COLD_START_RATIO (logged as COLD_START_THRESHOLD)

    Returns None if data is healthy.
    """
    try:
        rows = conn.execute(
            """
            SELECT DATE(ingested_ts) as day, COUNT(*) as cnt
            FROM news_feed
            WHERE DATE(ingested_ts) < DATE('now')
            GROUP BY day
            ORDER BY day DESC
            LIMIT ?
            """,
            (_DEGRADATION_LOOKBACK_DAYS,),
        ).fetchall()
    except Exception:
        return None

    if not rows:
        return None  # no history yet, skip check

    counts = [r[1] for r in rows]

    if len(counts) >= _DEGRADATION_LOOKBACK_DAYS:
        # Enough history: use P5 percentile
        counts_sorted = sorted(counts)
        p5_idx = max(0, int(len(counts_sorted) * 0.05) - 1)
        threshold = counts_sorted[p5_idx]
        threshold_label = f"P5={threshold} (90d)"
    else:
        # Cold start: 30-day average * ratio
        recent = counts[:30] if len(counts) >= 30 else counts
        avg = sum(recent) / len(recent)
        threshold = avg * _COLD_START_RATIO
        threshold_label = f"{_COLD_START_RATIO:.0%} of 30d avg={avg:.0f} [COLD_START_THRESHOLD]"

    if today_count < threshold:
        return (
            f"DATA_DEGRADED: today={today_count} < threshold {threshold_label}"
        )
    return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(
    db_path: str,
    symbols: Optional[List[str]] = None,
    max_per_symbol: int = 10,
) -> dict:
    conn = connect(db_path)
    ensure_news_tables(conn)

    if not symbols:
        symbols = get_tracked_symbols(conn)
    if not symbols:
        print("[ingester] No symbols found; pass --symbols or run db_update.py first.")
        conn.close()
        return {"symbols_processed": 0, "total_inserted": 0, "total_skipped": 0, "failed": []}

    ingested_ts = _now_utc()
    total_inserted = total_skipped = 0
    failed_symbols: List[str] = []
    sym_results: List[dict] = []

    print(f"[ingester] Starting — {len(symbols)} symbol(s), max {max_per_symbol} articles each")
    print(f"[ingester] Run time: {ingested_ts}")
    print()

    for i, sym in enumerate(symbols):
        # Rate limit: ≤ 1 req/sec between symbols (Governance Rule 3.2)
        if i > 0:
            time.sleep(_SYMBOL_INTERVAL)
        try:
            articles = fetch_news_for_symbol(sym, max_per_symbol=max_per_symbol)
            ins, skp = ingest_articles(conn, sym, articles, ingested_ts)
            total_inserted += ins
            total_skipped += skp
            sym_results.append({"symbol": sym, "inserted": ins, "skipped": skp, "ok": True})
            if ins > 0:
                print(f"  {sym}: +{ins} new, {skp} dup")
        except Exception as exc:
            failed_symbols.append(sym)
            sym_results.append({"symbol": sym, "inserted": 0, "skipped": 0, "ok": False, "error": str(exc)})
            print(f"  {sym}: FAILED — {exc}")

    # Degradation check
    degradation_warn = check_degradation(conn, total_inserted)
    conn.close()

    # -------------------------------------------------------------------------
    # Summary report
    # -------------------------------------------------------------------------
    print()
    print("=" * 55)
    print("[ingester] COLLECTION SUMMARY")
    print("=" * 55)
    print(f"  Run time    : {ingested_ts}")
    print(f"  Symbols     : {len(symbols)} total, {len(failed_symbols)} failed")
    print(f"  Articles    : {total_inserted} new, {total_skipped} duplicates skipped")

    # Top symbols by new articles
    top = sorted([r for r in sym_results if r["ok"] and r["inserted"] > 0],
                 key=lambda x: x["inserted"], reverse=True)[:5]
    if top:
        print(f"  Top symbols : " + ", ".join(f"{r['symbol']}(+{r['inserted']})" for r in top))

    if failed_symbols:
        print(f"  Failed      : {', '.join(failed_symbols)}")

    if degradation_warn:
        print(f"  ⚠  {degradation_warn}")
    else:
        print(f"  Health      : OK")
    print("=" * 55)

    return {
        "symbols_processed": len(symbols) - len(failed_symbols),
        "total_inserted": total_inserted,
        "total_skipped": total_skipped,
        "failed": failed_symbols,
        "degradation_warning": degradation_warn,
        "ingested_ts": ingested_ts,
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Fetch and persist news to SQLite news_feed table.")
    ap.add_argument("--db", default="japan_market.db", help="Path to SQLite DB")
    ap.add_argument(
        "--symbols",
        default="",
        help="Comma-separated list of symbols (default: all in daily_prices)",
    )
    ap.add_argument(
        "--max-per-symbol",
        type=int,
        default=10,
        dest="max_per_symbol",
        help="Max articles to fetch per symbol per run",
    )
    args = ap.parse_args()

    sym_list = [s.strip() for s in args.symbols.split(",") if s.strip()] or None
    run(db_path=args.db, symbols=sym_list, max_per_symbol=args.max_per_symbol)
