"""TDnet disclosure source parsers (P10-14 Cycle 1).

Two source formats supported:
  1. Yanoshin Web API JSON (https://webapi.yanoshin.jp/webapi/tdnet/...)
  2. TDnet raw HTML list (https://www.release.tdnet.info/inbs/I_list_001_*.html)

No network calls in this module — all parsing operates on already-fetched
payload. Network adapter lives in `tdnet_rss_adapter.py` (Cycle 2).

Category classification uses Japanese-keyword regex on disclosure title; order
matters because some titles match multiple categories (e.g., "公開買付けに関する
業績への影響" — TOB takes precedence over earnings).
"""
from __future__ import annotations

import re
from datetime import datetime
from typing import Any, Mapping, Sequence

from bs4 import BeautifulSoup

from hot_theme_rotator.data.external.tdnet_schema import (
    TdnetDisclosure,
    compute_disclosure_id,
)


class TdnetParseError(ValueError):
    """Raised when parser cannot extract required fields from source data."""


_CATEGORY_RULES: tuple[tuple[str, "re.Pattern[str]"], ...] = (
    ("tob",        re.compile(r"(公開買付|公開買い付け|TOB)")),
    ("split",      re.compile(r"(株式分割|株式併合|株式の分割|株式の併合)")),
    ("suspension", re.compile(r"(売買停止|上場廃止|監理銘柄)")),
    ("dividend",   re.compile(r"(配当|増配|減配|無配)")),
    ("order",      re.compile(r"(大型受注|業務提携|資本提携|子会社化|株式取得|株式の取得|事業提携)")),
    ("earnings",   re.compile(r"(業績|決算|通期予想|四半期)")),
    ("governance", re.compile(r"(役員|取締役|代表取締役|業務改善|改善命令|処分|是正)")),
)


def classify_category(title: str) -> str:
    """Map a Japanese disclosure title to one of ALLOWED_TDNET_CATEGORIES.

    Order-sensitive: TOB precedes earnings so that a title like
    "公開買付けに関する業績への影響" classifies as `tob`.
    Unmapped titles fall back to `other`.
    """
    if not isinstance(title, str):
        return "other"
    for category, pattern in _CATEGORY_RULES:
        if pattern.search(title):
            return category
    return "other"


def normalize_ticker(raw_code: str) -> str:
    """Normalize raw TDnet ticker code to 'NNNN.T' form.

    TDnet uses 5-digit codes (4-digit ticker + trailing '0' for ordinary
    equities). Some sources use plain 4-digit. Either input → 'NNNN.T'.
    """
    if not isinstance(raw_code, str):
        raise TdnetParseError(
            f"ticker code must be string, got {type(raw_code).__name__}"
        )
    code = raw_code.strip()
    if len(code) == 5 and code.isdigit() and code.endswith("0"):
        return f"{code[:4]}.T"
    if len(code) == 4 and code.isdigit():
        return f"{code}.T"
    raise TdnetParseError(f"unrecognized TDnet ticker code: {raw_code!r}")


def parse_yanoshin_json(
    payload: Mapping[str, Any],
    *,
    collected_ts: str,
) -> tuple[TdnetDisclosure, ...]:
    """Parse a Yanoshin Web API JSON response into TdnetDisclosure records.

    Real Yanoshin response shape (verified 2026-05-25 live smoke test):
        {
          "total_count": N,
          "items": [
            {"Tdnet": {
                "company_code": "20150",
                "pubdate": "2026-05-25 18:15:00",      # space-separated, no TZ
                "title": "...",
                "document_url": "https://...",
                "company_name": "...",
                ...
            }},
            ...
          ]
        }

    Also tolerates the older flat shape (for backwards-compatible fixtures and
    third-party callers): items may be `{"company_code": ..., "url": ..., ...}`
    directly without the `{"Tdnet": ...}` wrapper.

    Items with missing or malformed required fields are skipped silently —
    callers compare `len(input.items)` vs `len(output)` for metrics.
    """
    if not isinstance(payload, Mapping):
        raise TdnetParseError(
            f"payload must be a mapping, got {type(payload).__name__}"
        )
    items = payload.get("items")
    if items is None:
        raise TdnetParseError("payload missing 'items' key")
    if not isinstance(items, Sequence) or isinstance(items, (str, bytes)):
        raise TdnetParseError("payload['items'] must be a sequence")

    out: list[TdnetDisclosure] = []
    for raw in items:
        if not isinstance(raw, Mapping):
            continue
        # Real Yanoshin wraps each item: {"Tdnet": {...}}; unwrap if present.
        inner = raw.get("Tdnet") if "Tdnet" in raw else raw
        if not isinstance(inner, Mapping):
            continue
        disclosure = _build_disclosure_from_yanoshin_item(inner, collected_ts)
        if disclosure is not None:
            out.append(disclosure)
    return tuple(out)


def _normalize_pubdate(pubdate: str) -> str:
    """Normalize Yanoshin TDnet pubdate to ISO 8601 with JST offset.

    Yanoshin returns either:
      - 'YYYY-MM-DD HH:MM:SS' (real shape, space-separated, no TZ)
      - 'YYYY-MM-DD HH:MM:SS+09:00' (rare variant)
      - 'YYYY-MM-DDTHH:MM:SS+09:00' (legacy fixture, T-separated with TZ)

    Per Codex review 2026-05-25 (Rule 8.2 PIT enforcement): naive timestamps
    (no TZ) are dangerous because `datetime.fromisoformat` accepts them but
    `available_ts > decision_cutoff` comparison semantics break when comparing
    naive vs aware. Since Yanoshin disclosures are published on JST exchange
    schedule, this helper appends `+09:00` to ANY input missing a timezone
    suffix — both space-separated and T-separated naive forms.
    """
    if not pubdate:
        return pubdate

    # Detect if a TZ suffix is already present (after the date portion).
    has_tz = (
        pubdate.endswith("Z")
        or "+" in pubdate[10:]
        or "-" in pubdate[10:]
    )

    # T-separated form: append JST if naive, otherwise pass through.
    if "T" in pubdate:
        if has_tz:
            return pubdate
        return pubdate + "+09:00"

    # Space-separated form: convert to T-separated, then append JST if naive.
    if " " in pubdate and len(pubdate) >= 19:
        normalized = pubdate.replace(" ", "T", 1)
        if not has_tz:
            normalized += "+09:00"
        return normalized

    return pubdate


def _build_disclosure_from_yanoshin_item(
    raw: Mapping[str, Any],
    collected_ts: str,
) -> TdnetDisclosure | None:
    try:
        ticker = normalize_ticker(str(raw.get("company_code", "")))
        pubdate_raw = str(raw.get("pubdate", "")).strip()
        published_ts = _normalize_pubdate(pubdate_raw)
        title = str(raw.get("title", "")).strip()
        # Real Yanoshin uses `document_url`; older fixtures use `url`.
        url = str(raw.get("url", "") or raw.get("document_url", "")).strip()
        if not (published_ts and title and url):
            return None
        datetime.fromisoformat(published_ts)
        company_name_raw = raw.get("company_name")
        company_name = (
            str(company_name_raw).strip() or None
            if company_name_raw is not None
            else None
        )
        return TdnetDisclosure(
            disclosure_id=compute_disclosure_id(ticker, published_ts, title),
            ticker=ticker,
            published_ts=published_ts,
            collected_ts=collected_ts,
            title=title,
            category=classify_category(title),
            url=url,
            company_name=company_name,
            raw=dict(raw),
        )
    except (TdnetParseError, ValueError):
        return None


def parse_tdnet_html(
    html: str,
    *,
    trade_date: str,
    collected_ts: str,
) -> tuple[TdnetDisclosure, ...]:
    """Parse a TDnet raw HTML list page into TdnetDisclosure records.

    Expected structure: a `<table>` with rows of 5 `<td>`:
        time | code | company_name | title | <a href=...>link</a>

    Time is HH:MM JST; combined with `trade_date` and the +09:00 offset to build
    a fully-qualified ISO `published_ts`.

    Header rows (using `<th>`) yield 0 `<td>` and are skipped automatically.
    Rows with malformed time or missing fields are skipped silently.
    """
    if not isinstance(html, str):
        raise TdnetParseError(f"html must be string, got {type(html).__name__}")
    soup = BeautifulSoup(html, "html.parser")
    out: list[TdnetDisclosure] = []
    for row in soup.find_all("tr"):
        cells = row.find_all("td")
        if len(cells) < 5:
            continue
        disclosure = _build_disclosure_from_html_row(
            cells, trade_date, collected_ts
        )
        if disclosure is not None:
            out.append(disclosure)
    return tuple(out)


def _build_disclosure_from_html_row(
    cells: list,
    trade_date: str,
    collected_ts: str,
) -> TdnetDisclosure | None:
    time_str = cells[0].get_text(strip=True)
    code = cells[1].get_text(strip=True)
    company_name = cells[2].get_text(strip=True)
    title = cells[3].get_text(strip=True)
    link = cells[4].find("a")
    url = link.get("href", "").strip() if link else ""
    if not (time_str and code and title and url):
        return None
    try:
        ticker = normalize_ticker(code)
        published_ts = f"{trade_date}T{time_str}:00+09:00"
        datetime.fromisoformat(published_ts)
        return TdnetDisclosure(
            disclosure_id=compute_disclosure_id(ticker, published_ts, title),
            ticker=ticker,
            published_ts=published_ts,
            collected_ts=collected_ts,
            title=title,
            category=classify_category(title),
            url=url,
            company_name=company_name or None,
        )
    except (TdnetParseError, ValueError):
        return None
