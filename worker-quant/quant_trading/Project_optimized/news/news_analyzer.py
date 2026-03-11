"""news_analyzer.py

News analysis pipeline: event clustering + LLM sentiment scoring.

Reads unscored articles from news_feed, assigns event_cluster_id via
TF-IDF clustering, then calls Claude API to score each representative
article.  Results are written to news_sentiment.

Design references
-----------------
  design/NEWS_SYSTEM_DESIGN_v1.md  §4 (flow control), §5 (data flow)
  design/NEWS_GOVERNANCE_v1.md     Rule 2.1, 4.1, 4.2, 5.1-5.3

Governance compliance
---------------------
  Rule 2.1  : Any per-item failure is caught; scoring continues.
  Rule 4.1  : Extreme scores (|score| == 1.0) log the triggering prompt.
  Rule 4.2  : Never overwrite existing scores; insert new model_version only.
  Rule 5.1-3: event_cluster_id assigned by TF-IDF cosine clustering,
              written back to news_feed; ss7 de-duplication uses this field.

Usage
-----
  # from Project_optimized/ root:
  python news/news_analyzer.py --db japan_market.db
  python news/news_analyzer.py --db japan_market.db --batch 50 --model claude-haiku-4-5-20251001
  python news/news_analyzer.py --db japan_market.db --dry-run
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
import time
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Optional

# Allow running from any working directory
sys.path.insert(0, str(Path(__file__).parent.parent))
from trade_schema import connect, ensure_news_tables

# ---------------------------------------------------------------------------
# Optional dependencies — fail gracefully if missing
# ---------------------------------------------------------------------------
try:
    import anthropic as _anthropic
    _ANTHROPIC_AVAILABLE = True
except ImportError:
    _ANTHROPIC_AVAILABLE = False

try:
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    _SKLEARN_AVAILABLE = True
except ImportError:
    _SKLEARN_AVAILABLE = False

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_MODEL = "claude-haiku-4-5-20251001"
DEFAULT_BATCH = 50
CLUSTER_SIM_THRESHOLD = 0.85   # Governance Rule 5.1
CLUSTER_TIME_WINDOW_H = 4      # ±4 hours
TOKEN_BUCKET_RPM = 30          # Governance / Design §4.2: ≤ 30 req/min
SCORE_TIMEOUT_S = 30           # Design §4.2: auto-skip after 30s
EXTREME_SCORE_THRESHOLD = 0.99 # Governance Rule 4.1: log prompt for |score| near 1.0


# ---------------------------------------------------------------------------
# Token bucket (rate limiter)
# ---------------------------------------------------------------------------
class _TokenBucket:
    """Simple token bucket: refills at `rpm` tokens/minute."""

    def __init__(self, rpm: int = TOKEN_BUCKET_RPM) -> None:
        self._rate = rpm / 60.0   # tokens per second
        self._tokens = float(rpm)
        self._last = time.monotonic()

    def acquire(self) -> None:
        """Block until a token is available."""
        while True:
            now = time.monotonic()
            elapsed = now - self._last
            self._tokens = min(float(TOKEN_BUCKET_RPM), self._tokens + elapsed * self._rate)
            self._last = now
            if self._tokens >= 1.0:
                self._tokens -= 1.0
                return
            sleep_s = (1.0 - self._tokens) / self._rate
            time.sleep(sleep_s)


# ---------------------------------------------------------------------------
# LLM scoring prompt
# ---------------------------------------------------------------------------
_SYSTEM_PROMPT = """\
You are a quantitative finance analyst specializing in Japanese equity markets.
Analyze the news article title provided and return ONLY a JSON object with no
additional text, markdown, or explanation.

Required JSON schema:
{
  "sentiment": <float in [-1.0, 1.0]>,
  "urgency": <float in [0.0, 1.0]>,
  "expected_impact_days": <integer 1-30>,
  "reason": "<one sentence in English>"
}

Scoring guide:
  sentiment  : -1.0 = very bearish, 0.0 = neutral, +1.0 = very bullish
  urgency    : 0.0 = routine, 1.0 = market-moving
  expected_impact_days: how many trading days this news will influence the stock
"""

_USER_TEMPLATE = """\
Symbol: {symbol}
Title: {title}
Source: {source}
Published: {published_ts}
"""


def _call_llm(
    client,
    model: str,
    symbol: str,
    title: str,
    source: str,
    published_ts: str,
) -> tuple[dict | None, str]:
    """Call Claude API and return (parsed_result, raw_prompt_text).

    Returns (None, prompt) on failure so caller can degrade gracefully.
    """
    prompt = _USER_TEMPLATE.format(
        symbol=symbol,
        title=title,
        source=source,
        published_ts=published_ts,
    )
    try:
        msg = client.messages.create(
            model=model,
            max_tokens=256,
            system=_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
            timeout=SCORE_TIMEOUT_S,
        )
        raw = (msg.content[0].text or "").strip()
        # Strip markdown code fences if present
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        result = json.loads(raw)
        return result, prompt
    except Exception:
        return None, prompt


# ---------------------------------------------------------------------------
# Event clustering (Governance Rule 5.1-5.3)
# ---------------------------------------------------------------------------

def _assign_clusters(
    conn: sqlite3.Connection,
    lookback_hours: int = 48,
) -> int:
    """Assign event_cluster_id to unclustered news_feed rows.

    Groups by symbol, then clusters titles using TF-IDF cosine similarity
    within a ±4-hour time window.  Writes cluster IDs back to news_feed.

    Returns number of rows updated.
    """
    if not _SKLEARN_AVAILABLE:
        print("[analyzer] sklearn not available — skipping event clustering")
        return 0

    cutoff = (
        datetime.now(timezone.utc) - timedelta(hours=lookback_hours)
    ).strftime("%Y-%m-%dT%H:%M:%SZ")

    rows = conn.execute(
        """
        SELECT news_id, symbol, title, published_ts
        FROM news_feed
        WHERE event_cluster_id IS NULL
          AND ingested_ts >= ?
        ORDER BY symbol, published_ts
        """,
        (cutoff,),
    ).fetchall()

    if not rows:
        return 0

    # Group by symbol
    by_symbol: dict[str, list] = {}
    for news_id, symbol, title, pub_ts in rows:
        by_symbol.setdefault(symbol, []).append(
            {"news_id": news_id, "title": title or "", "pub_ts": pub_ts or ""}
        )

    total_updated = 0

    for symbol, items in by_symbol.items():
        if len(items) == 1:
            # Single item — assign its own cluster
            cid = str(uuid.uuid4())
            conn.execute(
                "UPDATE news_feed SET event_cluster_id = ? WHERE news_id = ?",
                (cid, items[0]["news_id"]),
            )
            total_updated += 1
            continue

        titles = [it["title"] for it in items]
        try:
            vec = TfidfVectorizer(
                analyzer="char_wb", ngram_range=(2, 4), min_df=1
            )
            tfidf = vec.fit_transform(titles)
            sim = cosine_similarity(tfidf)
        except Exception:
            # Fallback: each item gets its own cluster
            for it in items:
                conn.execute(
                    "UPDATE news_feed SET event_cluster_id = ? WHERE news_id = ?",
                    (str(uuid.uuid4()), it["news_id"]),
                )
            total_updated += len(items)
            continue

        # Parse timestamps for time-window check
        def _parse_ts(ts_str: str) -> Optional[datetime]:
            try:
                return datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
            except Exception:
                return None

        pub_dts = [_parse_ts(it["pub_ts"]) for it in items]
        n = len(items)
        assigned: list[Optional[str]] = [None] * n

        for i in range(n):
            if assigned[i] is not None:
                continue
            cid = str(uuid.uuid4())
            assigned[i] = cid
            for j in range(i + 1, n):
                if assigned[j] is not None:
                    continue
                # Time window check
                if pub_dts[i] and pub_dts[j]:
                    delta_h = abs(
                        (pub_dts[i] - pub_dts[j]).total_seconds() / 3600.0
                    )
                    if delta_h > CLUSTER_TIME_WINDOW_H:
                        continue
                if float(sim[i, j]) >= CLUSTER_SIM_THRESHOLD:
                    assigned[j] = cid

        for idx, it in enumerate(items):
            cid = assigned[idx] or str(uuid.uuid4())
            conn.execute(
                "UPDATE news_feed SET event_cluster_id = ? WHERE news_id = ?",
                (cid, it["news_id"]),
            )
            total_updated += 1

    conn.commit()
    return total_updated


# ---------------------------------------------------------------------------
# Fetch unscored batch (holdings-first priority)
# ---------------------------------------------------------------------------

def _fetch_unscored_batch(
    conn: sqlite3.Connection,
    model_version: str,
    batch_size: int,
) -> list[dict]:
    """Return up to batch_size unscored news_feed rows.

    Priority: symbols present in the latest screening_history first
    (proxy for current holdings), then remaining symbols by ingested_ts.
    Governance Design §4.2: only pull items not yet scored for this model_version.
    """
    # Get held symbols from most recent screening_history entry
    held_symbols: set[str] = set()
    try:
        latest_asof = conn.execute(
            "SELECT MAX(asof) FROM screening_history"
        ).fetchone()[0]
        if latest_asof:
            rows = conn.execute(
                "SELECT ticker FROM screening_history WHERE asof = ?",
                (latest_asof,),
            ).fetchall()
            held_symbols = {r[0] for r in rows}
    except Exception:
        pass

    already_scored = {
        r[0]
        for r in conn.execute(
            "SELECT news_id FROM news_sentiment WHERE model_version = ?",
            (model_version,),
        ).fetchall()
    }

    candidates = conn.execute(
        """
        SELECT news_id, symbol, title, source, published_ts, ingested_ts
        FROM news_feed
        WHERE event_cluster_id IS NOT NULL
        ORDER BY ingested_ts DESC
        LIMIT ?
        """,
        (batch_size * 4,),  # over-fetch to allow priority sort + filtering
    ).fetchall()

    held, rest = [], []
    for row in candidates:
        news_id = row[0]
        if news_id in already_scored:
            continue
        (held if row[1] in held_symbols else rest).append(row)

    selected = (held + rest)[:batch_size]
    return [
        {
            "news_id": r[0],
            "symbol": r[1],
            "title": r[2],
            "source": r[3],
            "published_ts": r[4],
            "ingested_ts": r[5],
        }
        for r in selected
    ]


# ---------------------------------------------------------------------------
# Write sentiment result
# ---------------------------------------------------------------------------

def _write_sentiment(
    conn: sqlite3.Connection,
    news_id: str,
    model_version: str,
    result: dict,
    scored_ts: str,
) -> None:
    """Insert one row into news_sentiment.  Governance Rule 4.2: no UPDATE."""
    conn.execute(
        """
        INSERT OR IGNORE INTO news_sentiment
            (news_id, model_version, sentiment_score, urgency,
             expected_impact_days, reason, scored_ts)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            news_id,
            model_version,
            float(result.get("sentiment", 0.0)),
            float(result.get("urgency", 0.0)),
            int(result.get("expected_impact_days", 3)),
            str(result.get("reason", ""))[:500],
            scored_ts,
        ),
    )


def _write_timeout(
    conn: sqlite3.Connection,
    news_id: str,
    model_version: str,
) -> None:
    """Mark a timed-out item so it's skipped next batch."""
    conn.execute(
        """
        INSERT OR IGNORE INTO news_sentiment
            (news_id, model_version, sentiment_score, urgency,
             expected_impact_days, reason, scored_ts)
        VALUES (?, ?, 0.0, 0.0, 3, 'TIMEOUT', 'TIMEOUT')
        """,
        (news_id, model_version),
    )


# ---------------------------------------------------------------------------
# Main run
# ---------------------------------------------------------------------------

def run(
    db_path: str,
    model_version: str = DEFAULT_MODEL,
    batch_size: int = DEFAULT_BATCH,
    dry_run: bool = False,
) -> dict:
    if not _ANTHROPIC_AVAILABLE and not dry_run:
        print("[analyzer] ERROR: 'anthropic' package not installed. Run: pip install anthropic")
        return {"error": "anthropic_not_installed"}

    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key and not dry_run:
        print("[analyzer] ERROR: ANTHROPIC_API_KEY env var not set.")
        return {"error": "no_api_key"}

    conn = connect(db_path)
    ensure_news_tables(conn)

    # Step 1: event clustering
    print("[analyzer] Step 1/2 — assigning event clusters...")
    clustered = _assign_clusters(conn)
    print(f"[analyzer]   {clustered} rows updated with event_cluster_id")

    # Step 2: LLM scoring
    print(f"[analyzer] Step 2/2 — scoring batch (max {batch_size}, model={model_version})...")
    batch = _fetch_unscored_batch(conn, model_version, batch_size)

    if not batch:
        print("[analyzer] No unscored items found.")
        conn.close()
        return {"clustered": clustered, "scored": 0, "skipped": 0, "errors": 0}

    if dry_run:
        print(f"[analyzer] DRY RUN — would score {len(batch)} items:")
        for item in batch[:5]:
            print(f"  {item['symbol']} | {item['title'][:80]}")
        if len(batch) > 5:
            print(f"  ... and {len(batch) - 5} more")
        conn.close()
        return {"clustered": clustered, "scored": 0, "skipped": len(batch), "errors": 0, "dry_run": True}

    client = _anthropic.Anthropic(api_key=api_key)
    bucket = _TokenBucket(rpm=TOKEN_BUCKET_RPM)

    scored = errors = 0
    scored_ts_now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    for item in batch:
        bucket.acquire()
        result, prompt = _call_llm(
            client,
            model=model_version,
            symbol=item["symbol"],
            title=item["title"] or "",
            source=item["source"] or "",
            published_ts=item["published_ts"] or "",
        )

        if result is None:
            _write_timeout(conn, item["news_id"], model_version)
            errors += 1
            print(f"  FAIL  {item['symbol']} | {(item['title'] or '')[:60]}")
        else:
            # Governance Rule 4.1: log extreme scores
            sentiment = float(result.get("sentiment", 0.0))
            if abs(sentiment) >= EXTREME_SCORE_THRESHOLD:
                print(
                    f"  [EXTREME] {item['symbol']} score={sentiment:+.2f} "
                    f"| prompt_hash={hash(prompt) & 0xFFFFFF:06x} "
                    f"| reason={result.get('reason', '')[:80]}"
                )

            _write_sentiment(conn, item["news_id"], model_version, result, scored_ts_now)
            scored += 1
            print(
                f"  OK    {item['symbol']} | sent={sentiment:+.2f} "
                f"urg={result.get('urgency', 0):.2f} "
                f"days={result.get('expected_impact_days', '?')} "
                f"| {(item['title'] or '')[:50]}"
            )

    conn.commit()
    conn.close()

    summary = {
        "clustered": clustered,
        "scored": scored,
        "errors": errors,
        "model_version": model_version,
    }
    print(
        f"[analyzer] Done — {clustered} clustered, "
        f"{scored} scored, {errors} errors/timeouts"
    )
    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Cluster news events and score sentiment via LLM."
    )
    ap.add_argument("--db", default="japan_market.db", help="Path to SQLite DB")
    ap.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Claude model ID (default: {DEFAULT_MODEL})",
    )
    ap.add_argument(
        "--batch",
        type=int,
        default=DEFAULT_BATCH,
        help=f"Max items to score per run (default: {DEFAULT_BATCH})",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        dest="dry_run",
        help="Cluster only, print what would be scored, skip LLM calls",
    )
    args = ap.parse_args()

    run(
        db_path=args.db,
        model_version=args.model,
        batch_size=args.batch,
        dry_run=args.dry_run,
    )
