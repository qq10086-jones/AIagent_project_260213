"""IPO 打新监控模块 — 轻量信息看板 + 3条硬规则 + 资金冲突检查。

Usage:
    from ipo_monitor import build_ipo_report
    report = build_ipo_report("japan_market.db", cfg)

设计原则: 不做评分模型（样本量不够），只做排雷 + 信息聚合。
"""

from __future__ import annotations

import re
import sqlite3
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from typing import Optional

# ---------------------------------------------------------------------------
# HTTP helper (与 news_to_db.py 同模式，独立实现避免循环 import)
# ---------------------------------------------------------------------------
_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)


def _get(url: str, timeout: int = 10) -> Optional[bytes]:
    try:
        req = urllib.request.Request(url, headers={"User-Agent": _UA})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.read()
    except Exception:
        return None


# ---------------------------------------------------------------------------
# IPO 日历抓取 — 96ut.com schedule page
# ---------------------------------------------------------------------------

def fetch_ipo_calendar() -> list[dict]:
    """从 96ut.com/ipo/schedule.php 抓取近期 IPO 日历。

    96ut schedule 表结构:
      [上場日, code, 銘柄名, 仮条件決定日, BB期間, 公募価格決定日, 申込期間, ステータス]

    Returns:
        list of dicts with keys: name, code, listing_date, bb_period,
        offering_price_date, status, source
    """
    url = "https://96ut.com/ipo/schedule.php"
    data = _get(url, timeout=15)
    if not data:
        print("[ipo_monitor] 96ut.com 不可达，尝试备选源...")
        return _fetch_ipo_from_google_news()

    html = data.decode("utf-8", errors="replace")

    # 用正则直接解析 <tr>/<td> — 比 HTMLParser 更可靠
    rows = re.findall(r"<tr[^>]*>(.*?)</tr>", html, re.DOTALL)
    results = []
    for row_html in rows:
        cells = re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", row_html, re.DOTALL)
        clean = [re.sub(r"<[^>]+>", "", c).strip() for c in cells]
        if len(clean) < 4:
            continue
        # 跳过表头
        if "上場日" in clean[0] or "銘柄" in clean[0]:
            continue

        ipo = _parse_schedule_row(clean)
        if ipo:
            results.append(ipo)

    if not results:
        return _fetch_ipo_from_google_news()

    # 补充详细数据（吸収金額、OR 等）from 96ut list page
    _enrich_from_list_page(results)

    return results


def _parse_schedule_row(row: list[str]) -> Optional[dict]:
    """解析 96ut schedule 表一行。

    列: [上場日, code, 銘柄名, 仮条件決定日, BB期間, 公募価格決定日, 申込期間, ステータス]
    """
    if len(row) < 3:
        return None

    listing_date = row[0]  # e.g. "4/24(金)"
    code = row[1] if len(row) > 1 else ""
    name = row[2] if len(row) > 2 else ""
    bb_period = row[4] if len(row) > 4 else ""
    price_date = row[5] if len(row) > 5 else ""
    status = row[7] if len(row) > 7 else ""

    if not name or len(name) < 2:
        return None

    return {
        "name": name,
        "code": code,
        "listing_date": listing_date,
        "bb_period": bb_period,
        "offering_price_date": price_date,
        "status": status,
        "market": "",
        "offering_price": 0,
        "absorption_amount_jpy": 0,
        "offering_ratio": 0.0,
        "source": "96ut.com",
    }


def _enrich_from_list_page(ipos: list[dict]) -> None:
    """从 96ut.com/ipo/list.php 补充吸収金額・OR・市場区分等详细数据。"""
    url = "https://96ut.com/ipo/list.php"
    data = _get(url, timeout=15)
    if not data:
        return

    html = data.decode("utf-8", errors="replace")
    rows = re.findall(r"<tr[^>]*>(.*?)</tr>", html, re.DOTALL)

    # 构建 code → detail mapping
    detail_map = {}
    for row_html in rows:
        cells = re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", row_html, re.DOTALL)
        clean = [re.sub(r"<[^>]+>", "", c).strip() for c in cells]
        if len(clean) < 5:
            continue
        # list.php 通常有: 銘柄名, コード, 市場, 公開価格, 初値, 騰落率, 吸収金額(億), OR(%) ...
        # 具体列顺序需要容错
        for i, cell in enumerate(clean):
            # 找到 code 列
            if re.match(r"^\d{4}[A-Z]?$", cell):
                detail = {"code": cell}
                # 扫描同行其他列
                for j, c in enumerate(clean):
                    if c in ("グロース", "プライム", "スタンダード"):
                        detail["market"] = c
                    elif "億" in c:
                        try:
                            detail["absorption_amount_jpy"] = int(float(re.sub(r"[^\d.]", "", c)) * 1e8)
                        except ValueError:
                            pass
                    elif c.endswith("%") and "." in c:
                        try:
                            v = float(c.replace("%", ""))
                            if 0 < v < 100:
                                detail["offering_ratio"] = v / 100.0
                        except ValueError:
                            pass
                    elif re.match(r"^[\d,]+$", c.replace(" ", "")):
                        try:
                            v = int(c.replace(",", "").replace(" ", ""))
                            if 100 < v < 100000:
                                detail["offering_price"] = v
                        except ValueError:
                            pass
                detail_map[cell] = detail
                break

    # Merge details into IPO list
    for ipo in ipos:
        code = ipo.get("code", "")
        if code in detail_map:
            d = detail_map[code]
            for k in ("market", "offering_price", "absorption_amount_jpy", "offering_ratio"):
                if d.get(k):
                    ipo[k] = d[k]


def _fetch_ipo_from_google_news() -> list[dict]:
    """96ut 不可达时，用 Google News RSS 兜底获取近期 IPO 信息。"""
    kw = "新規上場 OR IPO 上場承認 OR 東証上場 site:jpx.co.jp OR site:96ut.com"
    url = f"https://news.google.com/rss/search?q={urllib.parse.quote(kw)}&hl=ja&gl=JP&ceid=JP:ja"
    data = _get(url)
    if not data:
        return []
    try:
        root = ET.fromstring(data)
    except Exception:
        return []

    results = []
    for item in root.iter("item"):
        title = (item.findtext("title") or "").strip()
        pub = (item.findtext("pubDate") or "").strip()
        if not title:
            continue
        # 只保留明确含 IPO/上場 的标题
        if not any(k in title for k in ("上場", "IPO", "新規", "公開価格")):
            continue
        results.append({
            "name": title,
            "listing_date": pub[:10] if pub else "",
            "market": "",
            "offering_price": 0,
            "absorption_amount_jpy": 0,
            "offering_ratio": 0.0,
            "source": "google_news (incomplete)",
        })
    return results[:10]


# ---------------------------------------------------------------------------
# 3 条硬规则
# ---------------------------------------------------------------------------
def score_ipo_risk(ipo: dict, regime_score: float, cfg: dict | None = None) -> dict:
    """3 条硬规则评估 IPO 风险。

    Rules:
        1. 吸収金額 > 50億 → 回避
        2. オファリングレシオ > 30% → 回避
        3. regime_score < 0.30 → 全部暂停

    Returns:
        {verdict: "申購"|"回避"|"注意", flags: [...], regime_ok: bool}
    """
    c = cfg or {}
    max_abs = c.get("max_absorption_jpy", 5_000_000_000)
    max_or = c.get("max_offering_ratio", 0.30)
    min_regime = c.get("min_regime_score", 0.30)

    flags = []
    regime_ok = regime_score >= min_regime

    if not regime_ok:
        flags.append(f"⚠️ 熊市暂停打新 (regime={regime_score:.2f} < {min_regime})")

    absorption = ipo.get("absorption_amount_jpy", 0)
    if absorption > 0 and absorption > max_abs:
        flags.append(f"❌ 吸収金額過大 ({absorption/1e8:.0f}億 > {max_abs/1e8:.0f}億)")

    ratio = ipo.get("offering_ratio", 0.0)
    if ratio > 0 and ratio > max_or:
        flags.append(f"❌ OR過大 ({ratio:.0%} > {max_or:.0%})")

    # verdict
    if any("❌" in f for f in flags):
        verdict = "回避"
    elif not regime_ok:
        verdict = "注意"
    elif absorption == 0 and ratio == 0.0:
        verdict = "情報不足"  # 数据不全时不盲目推荐
    else:
        verdict = "申購"

    return {"verdict": verdict, "flags": flags, "regime_ok": regime_ok}


# ---------------------------------------------------------------------------
# 资金冲突检查
# ---------------------------------------------------------------------------
def check_capital_conflict(
    db_path: str, cash_needed: float, strategy_id: str = "sprint"
) -> dict:
    """检查当前是否有足够现金参与 IPO 申购。

    Args:
        db_path: SQLite DB path
        cash_needed: 本次申购需冻结的金额
        strategy_id: 策略 ID

    Returns:
        {available_cash, positions_value, can_apply, message}
    """
    conn = sqlite3.connect(db_path)

    # 最新 snapshot
    row = conn.execute(
        """SELECT cash, positions_value, nav FROM account_snapshots
           WHERE strategy_id = ? ORDER BY asof DESC, ts DESC LIMIT 1""",
        (strategy_id,),
    ).fetchone()
    conn.close()

    if not row:
        return {
            "available_cash": 0,
            "positions_value": 0,
            "can_apply": False,
            "message": "无账户快照数据",
        }

    cash, pos_val, nav = row
    can_apply = cash >= cash_needed

    if can_apply:
        msg = f"余力OK: 现金¥{cash:,.0f} >= 需要¥{cash_needed:,.0f}"
    else:
        msg = f"余力不足: 现金¥{cash:,.0f} < 需要¥{cash_needed:,.0f}"

    return {
        "available_cash": cash,
        "positions_value": pos_val,
        "can_apply": can_apply,
        "message": msg,
    }


# ---------------------------------------------------------------------------
# IPO 新闻查询
# ---------------------------------------------------------------------------
def get_ipo_news(db_path: str) -> list[dict]:
    """从 news_items 表查询近期 IPO 相关新闻。"""
    conn = sqlite3.connect(db_path)
    rows = conn.execute(
        """SELECT DISTINCT summary_cn, sentiment_score, published_at, source
           FROM news_items
           WHERE impact_category = 'IPO'
             AND published_at >= datetime('now', '-72 hours')
           ORDER BY published_at DESC LIMIT 15""",
    ).fetchall()
    conn.close()
    return [
        {"summary": r[0], "score": r[1], "published_at": r[2], "source": r[3]}
        for r in rows
    ]


# ---------------------------------------------------------------------------
# 主入口: 构建完整 IPO 报告
# ---------------------------------------------------------------------------
def build_ipo_report(db_path: str, cfg: dict | None = None) -> dict:
    """构建完整 IPO 监控报告。

    Args:
        db_path: SQLite DB path
        cfg: config.yaml 中 ipo_monitor 配置

    Returns:
        {
            "calendar": [...],      # 近期 IPO 日历
            "scored": [...],        # 每只 IPO + verdict/flags
            "capital": {...},       # 资金冲突检查
            "ipo_news": [...],      # 近期 IPO 新闻
            "regime_score": float,  # 当前 regime
            "generated_at": str,
        }
    """
    cfg = cfg or {}
    hard_rules = cfg.get("hard_rules", {})

    # 1. 获取 regime score
    regime_score = _get_current_regime(db_path)

    # 2. 抓取 IPO 日历
    calendar = fetch_ipo_calendar()

    # 3. 对每只 IPO 打标
    scored = []
    for ipo in calendar:
        risk = score_ipo_risk(ipo, regime_score, hard_rules)
        scored.append({**ipo, **risk})

    # 4. 资金冲突检查 (按典型 IPO 申购金额估算)
    typical_cost = 100_000  # 假设单手10万円
    capital = check_capital_conflict(db_path, typical_cost, cfg.get("strategy_id", "sprint"))

    # 5. IPO 新闻
    ipo_news = get_ipo_news(db_path)

    return {
        "calendar": calendar,
        "scored": scored,
        "capital": capital,
        "ipo_news": ipo_news,
        "regime_score": regime_score,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


def _get_current_regime(db_path: str) -> float:
    """从最新的 regime_diagnosis.json 或 DB 获取 regime score。"""
    import json
    from pathlib import Path

    # 优先读 regime_diagnosis.json（daily_run 已写入）
    diag_path = Path(db_path).parent / "reports" / "regime_diagnosis.json"
    if diag_path.exists():
        try:
            d = json.loads(diag_path.read_text(encoding="utf-8"))
            score = d.get("regime_score_v2_with_event") or d.get("regime_score_v2")
            if score is not None:
                return float(score)
        except Exception:
            pass

    # fallback: 从 cross_asset_snapshots 取最新 cross_asset_score 作近似
    try:
        conn = sqlite3.connect(db_path)
        row = conn.execute(
            "SELECT cross_asset_score FROM cross_asset_snapshots ORDER BY asof DESC LIMIT 1"
        ).fetchone()
        conn.close()
        if row and row[0] is not None:
            return float(row[0])
    except Exception:
        pass

    return 0.5  # 无数据时返回中性值


# ---------------------------------------------------------------------------
# CLI 入口
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import json
    report = build_ipo_report("japan_market.db")
    print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
