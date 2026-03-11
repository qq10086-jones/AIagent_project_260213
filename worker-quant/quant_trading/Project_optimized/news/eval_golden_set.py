"""eval_golden_set.py

Golden Set regression test for news sentiment scoring.

Evaluates a Claude model against a human-labeled set of Japanese financial
news headlines.  Computes Macro-F1 (bearish / neutral / bullish) and drift
vs the previous evaluation for the same model.  Writes results to the
`sentiment_model_eval` table.

P3 pass criteria (NEWS_SYSTEM_TASKS_v1.md):
  Macro-F1  >= 0.75
  Drift     <= 0.10  (absolute difference vs last recorded F1)

Usage
-----
  # from Project_optimized/ root:
  ANTHROPIC_API_KEY=sk-... python news/eval_golden_set.py --db japan_market.db
  python news/eval_golden_set.py --db japan_market.db --model claude-haiku-4-5-20251001
  python news/eval_golden_set.py --db japan_market.db --golden-set news/golden_set/golden_set_v1.csv
  python news/eval_golden_set.py --db japan_market.db --dry-run
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent))
from trade_schema import connect, ensure_news_tables

# ---------------------------------------------------------------------------
# Optional dependencies
# ---------------------------------------------------------------------------
try:
    import anthropic as _anthropic
    _ANTHROPIC_AVAILABLE = True
except ImportError:
    _ANTHROPIC_AVAILABLE = False

try:
    from sklearn.metrics import f1_score
    _SKLEARN_AVAILABLE = True
except ImportError:
    _SKLEARN_AVAILABLE = False

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_MODEL    = "claude-haiku-4-5-20251001"
DEFAULT_GOLDEN   = Path(__file__).parent / "golden_set" / "golden_set_v1.csv"
MACRO_F1_MIN     = 0.75    # P3 pass threshold
DRIFT_MAX        = 0.10    # P3 max allowed drift vs last eval
SCORE_TIMEOUT_S  = 30
MIN_GOLDEN_SIZE  = 200     # warn if golden set < this

# Sentiment score → class label thresholds
_BEARISH_CUTOFF = -0.1
_BULLISH_CUTOFF =  0.1

# LLM prompts (shared with news_analyzer for consistency)
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

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _score_to_label(score: float) -> str:
    if score < _BEARISH_CUTOFF:
        return "bearish"
    if score > _BULLISH_CUTOFF:
        return "bullish"
    return "neutral"


def _call_llm(client, model: str, row: dict) -> Optional[float]:
    """Return sentiment score float, or None on failure."""
    prompt = _USER_TEMPLATE.format(
        symbol=row.get("symbol", ""),
        title=row.get("title", ""),
        source=row.get("source", ""),
        published_ts=row.get("published_ts", ""),
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
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        result = json.loads(raw)
        return float(result["sentiment"])
    except Exception:
        return None


def _macro_f1_manual(y_true: list[str], y_pred: list[str]) -> float:
    """Compute Macro-F1 without sklearn (fallback)."""
    classes = ["bearish", "neutral", "bullish"]
    f1_sum = 0.0
    for cls in classes:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == cls and p == cls)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != cls and p == cls)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == cls and p != cls)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0 else 0.0
        )
        f1_sum += f1
    return f1_sum / len(classes)


def _compute_macro_f1(y_true: list[str], y_pred: list[str]) -> float:
    if _SKLEARN_AVAILABLE:
        return float(f1_score(y_true, y_pred, labels=["bearish", "neutral", "bullish"],
                              average="macro", zero_division=0))
    return _macro_f1_manual(y_true, y_pred)


def _load_golden_set(path: Path) -> list[dict]:
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def _get_last_f1(conn, model_version: str) -> Optional[float]:
    """Return the most recent macro_f1 for this model_version, or None."""
    row = conn.execute(
        """
        SELECT macro_f1 FROM sentiment_model_eval
        WHERE model_version = ?
        ORDER BY eval_date DESC
        LIMIT 1
        """,
        (model_version,),
    ).fetchone()
    return float(row[0]) if row else None


def _write_eval_result(
    conn,
    model_version: str,
    golden_set_size: int,
    macro_f1: float,
    mean_abs_drift: float,
    passed: bool,
    detail: dict,
) -> None:
    eval_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    conn.execute(
        """
        INSERT OR IGNORE INTO sentiment_model_eval
            (model_version, eval_date, golden_set_size, macro_f1,
             mean_abs_drift, passed, eval_detail_json)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            model_version,
            eval_date,
            golden_set_size,
            macro_f1,
            mean_abs_drift,
            int(passed),
            json.dumps(detail, ensure_ascii=False),
        ),
    )
    conn.commit()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(
    db_path: str,
    model_version: str = DEFAULT_MODEL,
    golden_set_path: Optional[str] = None,
    dry_run: bool = False,
) -> dict:
    golden_path = Path(golden_set_path) if golden_set_path else DEFAULT_GOLDEN
    if not golden_path.exists():
        print(f"[eval] ERROR: golden set not found: {golden_path}")
        return {"error": "golden_set_not_found"}

    rows = _load_golden_set(golden_path)
    if not rows:
        print("[eval] ERROR: golden set is empty.")
        return {"error": "empty_golden_set"}

    if len(rows) < MIN_GOLDEN_SIZE:
        print(
            f"[eval] WARN: golden set has {len(rows)} rows "
            f"(minimum recommended: {MIN_GOLDEN_SIZE}). "
            "Results may not be statistically reliable."
        )

    print(f"[eval] Golden set: {len(rows)} items | model: {model_version}")
    print(f"[eval] Golden set path: {golden_path}")

    if dry_run:
        print("[eval] DRY RUN — listing first 5 items:")
        for r in rows[:5]:
            print(f"  {r.get('symbol','?')} | {r.get('sentiment_label','?')} | {r.get('title','')[:70]}")
        if len(rows) > 5:
            print(f"  ... and {len(rows) - 5} more")
        return {"dry_run": True, "golden_set_size": len(rows)}

    if not _ANTHROPIC_AVAILABLE:
        print("[eval] ERROR: 'anthropic' package not installed.")
        return {"error": "anthropic_not_installed"}

    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        print("[eval] ERROR: ANTHROPIC_API_KEY not set.")
        return {"error": "no_api_key"}

    conn = connect(db_path)
    ensure_news_tables(conn)
    last_f1 = _get_last_f1(conn, model_version)

    client = _anthropic.Anthropic(api_key=api_key)

    y_true: list[str] = []
    y_pred: list[str] = []
    failures = 0
    per_item: list[dict] = []

    # Simple rate limit: ~20 req/min (conservative for eval)
    _sleep_between = 60.0 / 20.0

    print(f"[eval] Scoring {len(rows)} items (rate-limited, ~20 req/min)...")
    for i, row in enumerate(rows):
        if i > 0:
            time.sleep(_sleep_between)

        score = _call_llm(client, model_version, row)
        gt_label = (row.get("sentiment_label") or "neutral").strip().lower()

        if score is None:
            failures += 1
            pred_label = "neutral"  # conservative fallback for failures
            print(f"  FAIL {row.get('news_id','?')} | {row.get('title','')[:50]}")
        else:
            pred_label = _score_to_label(score)
            match = "OK" if pred_label == gt_label else "MISMATCH"
            print(
                f"  {match} {row.get('news_id','?')} "
                f"gt={gt_label} pred={pred_label}(score={score:+.2f}) "
                f"| {row.get('title','')[:45]}"
            )

        y_true.append(gt_label)
        y_pred.append(pred_label)
        per_item.append({
            "news_id": row.get("news_id"),
            "symbol": row.get("symbol"),
            "gt": gt_label,
            "pred": pred_label,
            "score": score,
        })

    macro_f1 = _compute_macro_f1(y_true, y_pred)

    # Drift: absolute difference vs last recorded F1
    if last_f1 is not None:
        mean_abs_drift = abs(macro_f1 - last_f1)
    else:
        mean_abs_drift = 0.0  # no prior — no drift penalty

    passed = (macro_f1 >= MACRO_F1_MIN) and (mean_abs_drift <= DRIFT_MAX)

    detail = {
        "macro_f1": macro_f1,
        "mean_abs_drift": mean_abs_drift,
        "last_f1": last_f1,
        "failures": failures,
        "per_item_count": len(per_item),
        "confusion": _confusion_summary(y_true, y_pred),
    }

    _write_eval_result(
        conn, model_version, len(rows), macro_f1, mean_abs_drift, passed, detail
    )
    conn.close()

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    print()
    print("=" * 55)
    print("[eval] REGRESSION RESULT")
    print("=" * 55)
    print(f"  Model         : {model_version}")
    print(f"  Golden set    : {len(rows)} items ({failures} API failures)")
    print(f"  Macro-F1      : {macro_f1:.4f}  (threshold >= {MACRO_F1_MIN})")
    if last_f1 is not None:
        print(f"  Drift         : {mean_abs_drift:.4f}  (threshold <= {DRIFT_MAX})  [last: {last_f1:.4f}]")
    else:
        print(f"  Drift         : N/A (first evaluation for this model)")
    print(f"  Result        : {'PASS' if passed else 'FAIL'}")
    print("=" * 55)

    return {
        "model_version": model_version,
        "golden_set_size": len(rows),
        "macro_f1": macro_f1,
        "mean_abs_drift": mean_abs_drift,
        "passed": passed,
        "failures": failures,
    }


def _confusion_summary(y_true: list[str], y_pred: list[str]) -> dict:
    classes = ["bearish", "neutral", "bullish"]
    matrix: dict[str, dict[str, int]] = {
        c: {"bearish": 0, "neutral": 0, "bullish": 0} for c in classes
    }
    for t, p in zip(y_true, y_pred):
        if t in matrix and p in matrix[t]:
            matrix[t][p] += 1
    return matrix


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Run golden set regression test for news sentiment scoring."
    )
    ap.add_argument("--db", default="japan_market.db", help="Path to SQLite DB")
    ap.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Claude model ID (default: {DEFAULT_MODEL})",
    )
    ap.add_argument(
        "--golden-set",
        dest="golden_set",
        default=None,
        help=f"Path to golden set CSV (default: {DEFAULT_GOLDEN})",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        dest="dry_run",
        help="List golden set items without calling LLM",
    )
    args = ap.parse_args()

    result = run(
        db_path=args.db,
        model_version=args.model,
        golden_set_path=args.golden_set,
        dry_run=args.dry_run,
    )
    sys.exit(0 if result.get("passed", True) else 1)
