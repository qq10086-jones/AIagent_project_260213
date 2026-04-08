"""宏观情报 LLM 分析管线 — 层3: Gemma 4 27B via Ollama API。

仅在层2 (macro_event_detector) 触发 L1/L2 事件时调用。
读取跨资产数据 + 新闻标题 → 调 Ollama → 输出事件分类/因果链/regime_boost。
LLM 超时或失败时退化到层2规则引擎的 boost 值。
"""
from __future__ import annotations

import argparse
import json
import re
import sqlite3
import urllib.request
import urllib.error
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

JST = timezone(timedelta(hours=9))

# ── 新闻标题采集 ──────────────────────────────────────────────────

def ensure_macro_headlines_table(conn: sqlite3.Connection) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS macro_headlines (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            asof TEXT NOT NULL,
            title TEXT NOT NULL,
            source TEXT,
            url TEXT,
            fetched_ts TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()


def fetch_macro_headlines(
    asof: str,
    queries: List[str] | None = None,
    max_per_query: int = 5,
) -> List[Dict[str, str]]:
    """Google News RSS で宏観ヘッドラインを取得。

    Returns list of {"title": ..., "source": "google_news", "url": ...}
    """
    if queries is None:
        queries = [
            "日経平均",
            "Japan stock market",
            "oil price crude",
            "Federal Reserve BOJ",
        ]

    headlines: List[Dict[str, str]] = []

    for query in queries:
        try:
            import xml.etree.ElementTree as ET
            encoded_q = urllib.request.quote(query)
            url = f"https://news.google.com/rss/search?q={encoded_q}&hl=ja&gl=JP&ceid=JP:ja"
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = resp.read().decode("utf-8", errors="replace")
            root = ET.fromstring(data)
            items = root.findall(".//item")[:max_per_query]
            for item in items:
                title_el = item.find("title")
                link_el = item.find("link")
                if title_el is not None and title_el.text:
                    headlines.append({
                        "title": title_el.text.strip(),
                        "source": "google_news",
                        "url": link_el.text.strip() if link_el is not None and link_el.text else "",
                        "query": query,
                    })
        except Exception as e:
            print(f"[macro_digest] headline fetch failed for '{query}': {e}")

    return headlines


def save_macro_headlines(
    conn: sqlite3.Connection, asof: str, headlines: List[Dict[str, str]]
) -> int:
    """見出しを DB に保存。"""
    ensure_macro_headlines_table(conn)
    saved = 0
    for h in headlines:
        try:
            conn.execute(
                "INSERT INTO macro_headlines (asof, title, source, url) VALUES (?,?,?,?)",
                (asof, h["title"], h.get("source", ""), h.get("url", "")),
            )
            saved += 1
        except Exception:
            pass
    conn.commit()
    return saved


# ── LLM 呼び出し ─────────────────────────────────────────────────

def build_llm_prompt(
    snapshot: Dict[str, Any],
    headlines: List[Dict[str, str]],
    alert_level: str,
    triggered_rules: List[Dict[str, Any]] | None = None,
) -> str:
    """LLM に送る prompt を構築。"""

    def _fmt(key: str, default: str = "N/A") -> str:
        v = snapshot.get(key)
        if v is None:
            return default
        return f"{float(v):+.2f}%"

    rules_str = ""
    if triggered_rules:
        rules_str = ", ".join(r["name"] for r in triggered_rules)

    headlines_str = ""
    if headlines:
        for i, h in enumerate(headlines[:15], 1):
            headlines_str += f"  {i}. {h['title']}\n"
    else:
        headlines_str = "  (新聞ヘッドライン取得不可)\n"

    prompt = f"""你是日本股市宏観分析師。以下是今日盤前市場数据:

CME日経期貨: {_fmt('nk_futures_gap_pct')} | WTI原油: {_fmt('crude_oil_change_pct')} | VIX: {_fmt('vix_change_pct')}
USD/JPY: {_fmt('usdjpy_change_pct')} | S&P500: {_fmt('sp500_overnight_pct')} | SOX: {_fmt('sox_change_pct')}
黄金: {_fmt('gold_change_pct')} | 銅: {_fmt('copper_change_pct')}

規則引擎判定: alert_level={alert_level}, 触発規則: {rules_str or '(无)'}

相関新聞標題:
{headlines_str}
任务:
1. 判断最可能的事件原因（一句话，20字以内）
2. 事件類型: geopolitical / monetary_policy / trade_policy / commodity_shock / earnings_macro / other
3. 対日経的影響: positive / negative / mixed
4. regime_boost 建議: -0.30 到 +0.30 的浮点数
5. 影響持続天数: 1-5 的整数
6. 受益板塊和受損板塊

只輸出 JSON:
{{
  "event_summary": "",
  "event_type": "",
  "impact_direction": "",
  "regime_boost": 0.0,
  "duration_days": 3,
  "sectors_positive": [],
  "sectors_negative": [],
  "confidence": 0.0
}}"""
    return prompt


def call_llm_analysis(
    prompt: str,
    endpoint: str = "http://localhost:11434",
    model: str = "gemma4:26b",
    timeout: int = 60,
) -> Optional[str]:
    """Ollama chat API を叩いて LLM 応答を取得。
    /api/chat を優先使用（gemma4 では /api/generate が空レスポンスを返すことがある）。
    """
    url = f"{endpoint}/api/chat"
    payload = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "options": {
            "temperature": 0.3,
            "num_predict": 2048,  # gemma4 uses thinking tokens before output; 512 is insufficient
        },
    }).encode("utf-8")

    req = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            result = json.loads(resp.read().decode("utf-8"))
            # chat API: result["message"]["content"]
            msg = result.get("message") or {}
            content = msg.get("content", "")
            if not content:
                # fallback: some builds return top-level "response"
                content = result.get("response", "")
            return content or None
    except urllib.error.URLError as e:
        print(f"[macro_digest] Ollama request failed: {e}")
        return None
    except Exception as e:
        print(f"[macro_digest] LLM call error: {e}")
        return None


def parse_llm_response(raw: str) -> Optional[Dict[str, Any]]:
    """LLM 応答から JSON を抽出してバリデーション。"""
    if not raw:
        return None

    # 1. ```json ... ``` ブロック
    json_match = re.search(r'```json\s*(.*?)\s*```', raw, re.DOTALL)
    if json_match:
        json_str = json_match.group(1)
    else:
        # 2. 最初の { から最後の } まで（ネスト対応）
        start = raw.find('{')
        end = raw.rfind('}')
        if start != -1 and end != -1 and end > start:
            json_str = raw[start:end + 1]
        else:
            return None

    try:
        data = json.loads(json_str)
    except json.JSONDecodeError:
        return None

    # バリデーション
    required = ["event_summary", "event_type", "impact_direction", "regime_boost", "duration_days"]
    for key in required:
        if key not in data:
            return None

    # clamp regime_boost
    boost = float(data.get("regime_boost", 0.0))
    data["regime_boost"] = max(-0.30, min(0.30, boost))

    # clamp duration
    dur = int(data.get("duration_days", 3))
    data["duration_days"] = max(1, min(5, dur))

    # confidence check
    conf = float(data.get("confidence", 0.5))
    data["confidence"] = max(0.0, min(1.0, conf))

    # confidence < 0.5 → boost 半減
    if data["confidence"] < 0.5:
        data["regime_boost"] = data["regime_boost"] * 0.5
        data["confidence_adjusted"] = True
    else:
        data["confidence_adjusted"] = False

    # 有効な event_type チェック
    valid_types = {"geopolitical", "monetary_policy", "trade_policy",
                   "commodity_shock", "earnings_macro", "other"}
    if data.get("event_type") not in valid_types:
        data["event_type"] = "other"

    return data


def update_macro_event_with_llm(
    conn: sqlite3.Connection,
    asof: str,
    llm_result: Dict[str, Any],
) -> None:
    """macro_events テーブルの既存行に LLM 分析結果を上書き。"""
    llm_boost = float(llm_result.get("regime_boost", 0.0))

    # final_boost = LLM が分析に成功した場合は LLM の boost を採用
    # (規則引擎の boost との平均ではなく、LLM を優先)
    conn.execute(
        """UPDATE macro_events SET
           llm_boost = ?,
           final_boost = ?,
           duration_days = ?,
           event_summary = ?,
           event_type = ?,
           llm_raw_json = ?,
           source = 'llm'
           WHERE asof = ?""",
        (
            llm_boost,
            llm_boost,  # LLM 成功時は LLM boost を final とする
            int(llm_result.get("duration_days", 3)),
            str(llm_result.get("event_summary", "")),
            str(llm_result.get("event_type", "other")),
            json.dumps(llm_result, ensure_ascii=False),
            asof,
        ),
    )
    conn.commit()


# ── メインパイプライン ────────────────────────────────────────────

def run_macro_digest(
    conn: sqlite3.Connection,
    asof: str,
    snapshot: Dict[str, Any],
    alert_level: str,
    triggered_rules: List[Dict[str, Any]] | None = None,
    *,
    endpoint: str = "http://localhost:11434",
    model: str = "gemma4:27b",
    timeout: int = 60,
) -> Dict[str, Any]:
    """宏観情報パイプライン全体を実行。

    Returns:
        {"status": "llm_success" / "llm_failed" / "skipped",
         "llm_result": {...} or None}
    """
    if alert_level not in ("L1", "L2"):
        return {"status": "skipped", "reason": "alert_level is none/L3", "llm_result": None}

    # 1. ヘッドライン取得
    print(f"[macro_digest] Fetching headlines for {alert_level} event...")
    headlines = fetch_macro_headlines(asof)
    saved = save_macro_headlines(conn, asof, headlines)
    print(f"[macro_digest] Saved {saved} headlines")

    # 2. Prompt 構築
    prompt = build_llm_prompt(snapshot, headlines, alert_level, triggered_rules)

    # 3. LLM 呼び出し
    print(f"[macro_digest] Calling {model} at {endpoint}...")
    raw_response = call_llm_analysis(prompt, endpoint=endpoint, model=model, timeout=timeout)

    if raw_response is None:
        print("[macro_digest] LLM call failed — falling back to rule engine boost")
        return {"status": "llm_failed", "reason": "request_failed", "llm_result": None}

    # 4. Parse + validate
    llm_result = parse_llm_response(raw_response)
    if llm_result is None:
        print(f"[macro_digest] LLM response parse failed — falling back to rule engine boost")
        print(f"[macro_digest] Raw response: {raw_response[:200]}...")
        return {"status": "llm_failed", "reason": "parse_failed", "llm_result": None, "raw": raw_response}

    # 5. DB 更新
    update_macro_event_with_llm(conn, asof, llm_result)
    print(f"[macro_digest] LLM analysis: {llm_result.get('event_summary', 'N/A')}")
    print(f"[macro_digest] regime_boost={llm_result['regime_boost']:+.4f}  "
          f"duration={llm_result['duration_days']}d  confidence={llm_result.get('confidence', 0):.2f}")

    return {"status": "llm_success", "llm_result": llm_result}


# ── CLI ──────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="宏観情報 LLM 分析パイプライン")
    ap.add_argument("--db", default="japan_market.db")
    ap.add_argument("--endpoint", default="http://localhost:11434",
                    help="Ollama API endpoint (e.g. http://gemma-host:11434)")
    ap.add_argument("--model", default="gemma4:26b")
    ap.add_argument("--timeout", type=int, default=60)
    ap.add_argument("--asof", default=None)
    args = ap.parse_args()

    conn = sqlite3.connect(args.db)

    # 読取最新 cross_asset_snapshot
    from cross_asset_signals import load_latest_cross_asset
    snapshot = load_latest_cross_asset(conn, asof=args.asof)
    if not snapshot:
        print("[macro_digest] No cross_asset_snapshot found")
        conn.close()
        return

    asof = snapshot.get("asof", args.asof or datetime.now(JST).strftime("%Y-%m-%d"))

    # 読取最新 macro_event
    from macro_event_detector import load_active_event, detect_macro_events
    active = load_active_event(conn, asof)

    if active is None:
        # 実行時に snapshot から直接検出
        result = detect_macro_events(snapshot)
        alert_level = result.get("alert_level", "none")
        triggered_rules = result.get("triggered_rules", [])
    else:
        alert_level = active.get("alert_level", "none")
        triggered_rules = []  # 既存イベントの場合はルール詳細は不要

    if alert_level not in ("L1", "L2"):
        print(f"[macro_digest] alert_level={alert_level} — LLM analysis not needed")
        conn.close()
        return

    digest_result = run_macro_digest(
        conn, asof, snapshot, alert_level, triggered_rules,
        endpoint=args.endpoint,
        model=args.model,
        timeout=args.timeout,
    )

    print(f"[macro_digest] Status: {digest_result['status']}")
    conn.close()


if __name__ == "__main__":
    main()
