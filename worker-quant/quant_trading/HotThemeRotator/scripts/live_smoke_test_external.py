"""Live smoke test for P10-14 / P10-19 / P10-16 external adapters.

Hits REAL endpoints to surface URL / selector / schema mismatches that mock-HTTP
tests cannot find. Sources requiring credentials (TwelveData, J-Quants) are
SKIPPED automatically when env vars are missing.

Run from project root:
    python scripts/live_smoke_test_external.py

Exits 0 if all non-skipped sources work. Non-zero count of failures otherwise.
"""
from __future__ import annotations

import sys
import time
from datetime import date, timedelta
from pathlib import Path


# Windows cp932 stdout cannot encode ¥ or CJK — force UTF-8 like morning_briefing.py
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

import requests

from hot_theme_rotator.data.external.jquants_live_bridge import (
    JquantsAuthError,
    JquantsCredentials,
    JquantsFetchError,
    JquantsLiveBridge,
)
from hot_theme_rotator.data.external.realtime_price.kabutan_scraper import (
    KabutanParseError,
    kabutan_url,
    parse_kabutan_html,
)
from hot_theme_rotator.data.external.realtime_price.stooq_csv_fetcher import (
    StooqParseError,
    parse_stooq_csv,
    stooq_url,
)
from hot_theme_rotator.data.external.realtime_price.twelvedata_client import (
    TwelveDataError,
    get_api_key_from_env,
    parse_twelvedata_response,
    twelvedata_url,
)
from hot_theme_rotator.data.external.realtime_price.yahoo_japan_scraper import (
    YahooJapanParseError,
    parse_yahoo_japan_html,
    yahoo_japan_url,
)
from hot_theme_rotator.data.external.tdnet_rss_adapter import (
    TdnetFetchError,
    YanoshinTdnetAdapter,
)


UA = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0 Safari/537.36 HotThemeRotator/1.0"
    )
}
TIMEOUT = 20

results: list[tuple[str, str, str]] = []


def banner(title: str) -> None:
    print(f"\n{'=' * 64}\n {title}\n{'=' * 64}")


def record(name: str, status: str, detail: str) -> None:
    print(f"  [{status}] {detail}")
    results.append((name, status, detail))


def safe(name: str, fn) -> None:
    banner(name)
    t0 = time.time()
    try:
        result = fn()
        if result and str(result).startswith("SKIP"):
            record(name, "SKIP", str(result))
        else:
            record(name, "PASS", f"{result} ({time.time() - t0:.1f}s)")
    except Exception as exc:  # noqa: BLE001
        record(name, "FAIL", f"{type(exc).__name__}: {str(exc)[:200]}")


# ---------------------------------------------------------------
# 1. Yanoshin TDnet
def test_yanoshin():
    today = date.today().isoformat()
    print(f"  date: {today}")
    adapter = YanoshinTdnetAdapter()
    records = adapter.fetch_list_for_date(today, limit=5)
    return f"got {len(records)} disclosures for {today}"


# 2. Yahoo Finance Japan
def test_yahoo_jp():
    url = yahoo_japan_url("6779.T")
    print(f"  URL: {url}")
    r = requests.get(url, headers=UA, timeout=TIMEOUT)
    print(f"  status: {r.status_code}, body length: {len(r.text)}")
    if r.status_code != 200:
        raise RuntimeError(f"HTTP {r.status_code}")
    q = parse_yahoo_japan_html(r.text, symbol="6779.T")
    return f"price ¥{q.price}"


# 3. Kabutan
def test_kabutan():
    url = kabutan_url("6779.T")
    print(f"  URL: {url}")
    r = requests.get(url, headers=UA, timeout=TIMEOUT)
    print(f"  status: {r.status_code}, body length: {len(r.text)}")
    if r.status_code != 200:
        raise RuntimeError(f"HTTP {r.status_code}")
    q = parse_kabutan_html(r.text, symbol="6779.T")
    return f"price ¥{q.price}"


# 4. Stooq
def test_stooq():
    url = stooq_url("6779.T")
    print(f"  URL: {url}")
    r = requests.get(url, headers=UA, timeout=TIMEOUT)
    print(f"  status: {r.status_code}, body length: {len(r.text)}")
    if r.status_code != 200:
        raise RuntimeError(f"HTTP {r.status_code}")
    print(f"  first 300 chars: {r.text[:300]!r}")
    q = parse_stooq_csv(r.text, symbol="6779.T")
    return f"latest close ¥{q.price} as of {q.data_ts}"


# 5. TwelveData (skip if no API key)
def test_twelvedata():
    try:
        api_key = get_api_key_from_env()
    except TwelveDataError as exc:
        return f"SKIP: {exc}"
    url = twelvedata_url("6779.T", api_key=api_key)
    print(f"  URL: {url[:80]}...(key hidden)")
    r = requests.get(url, headers=UA, timeout=TIMEOUT)
    print(f"  status: {r.status_code}, body length: {len(r.text)}")
    if r.status_code != 200:
        raise RuntimeError(f"HTTP {r.status_code}: {r.text[:200]}")
    q = parse_twelvedata_response(r.text, symbol="6779.T")
    return f"price ¥{q.price}"


# 6. J-Quants (skip if no credentials)
def test_jquants():
    creds = JquantsCredentials.from_env()
    if not (creds.refresh_token or (creds.email and creds.password)):
        return "SKIP: no JQUANTS_REFRESH_TOKEN or JQUANTS_EMAIL+PASSWORD env"
    bridge = JquantsLiveBridge(credentials=creds)
    today = date.today()
    week_ago = (today - timedelta(days=7)).isoformat()
    bars = bridge.fetch_daily_quotes("1306.T", date_from=week_ago)
    if bars:
        return f"got {len(bars)} bars for 1306.T; latest close ¥{bars[-1].close} on {bars[-1].asof}"
    return f"got 0 bars for 1306.T from {week_ago}"


def main() -> int:
    safe("Yanoshin TDnet (P10-14)", test_yanoshin)
    safe("Yahoo Finance Japan (P10-19)", test_yahoo_jp)
    safe("Kabutan (P10-19)", test_kabutan)
    safe("Stooq (P10-19)", test_stooq)
    safe("TwelveData (P10-19, needs API key)", test_twelvedata)
    safe("J-Quants (P10-16, needs credentials)", test_jquants)

    banner("SUMMARY")
    pass_n = sum(1 for _, s, _ in results if s == "PASS")
    fail_n = sum(1 for _, s, _ in results if s == "FAIL")
    skip_n = sum(1 for _, s, _ in results if s == "SKIP")
    for name, status, detail in results:
        print(f"  [{status}] {name}")
        if status != "PASS":
            print(f"         {detail[:150]}")
    print(f"\n  Total: {pass_n} PASS / {fail_n} FAIL / {skip_n} SKIP")
    return fail_n


if __name__ == "__main__":
    sys.exit(main())
