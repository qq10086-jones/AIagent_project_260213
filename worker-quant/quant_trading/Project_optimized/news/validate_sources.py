"""validate_sources.py

One-shot diagnostic for P2 data sources (TDnet RSS and J-Quants API).

Fetches real responses and prints the actual structure so the maintainer
can verify that news_ingester.py's parsing matches the live feed format.

This script is NOT part of the automated test suite — it requires real
network access and optionally a JQUANTS_REFRESH_TOKEN.  Run it manually
whenever TDnet or J-Quants changes their API structure.

Usage
-----
  # from Project_optimized/ root:
  python news/validate_sources.py
  python news/validate_sources.py --jquants-token <refresh_token>
  python news/validate_sources.py --symbol 9432
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import requests
except ImportError:
    print("ERROR: 'requests' not installed. Run: pip install requests")
    sys.exit(1)

_USER_AGENT   = "Mozilla/5.0 (compatible; QuantNewsBot/1.0)"
_TIMEOUT      = 20
_TDNET_URL    = "https://www.release.tdnet.info/inbs/I_rss.rdf"
_JQUANTS_BASE = "https://api.jquants.com/v1"

# Expected fields — update these if the API changes
_TDNET_EXPECTED_ITEM_FIELDS = {
    "namespaced_rdf_item": "{http://www.w3.org/1999/02/22-rdf-syntax-ns#}item",
    "title_dc":            "{http://purl.org/dc/elements/1.1/}title",
    "date_dc":             "{http://purl.org/dc/elements/1.1/}date",
    "link_attr":           "{http://www.w3.org/1999/02/22-rdf-syntax-ns#}about",
}
_JQUANTS_EXPECTED_KEYS = {"announcement", "CompanyName", "DocumentType", "Date", "URL"}


# ---------------------------------------------------------------------------
# TDnet validation
# ---------------------------------------------------------------------------

def validate_tdnet(verbose: bool = False) -> bool:
    print("\n[TDnet] Fetching:", _TDNET_URL)
    try:
        resp = requests.get(_TDNET_URL, timeout=_TIMEOUT,
                            headers={"User-Agent": _USER_AGENT})
    except Exception as e:
        print(f"  ERROR: {e}")
        return False

    print(f"  HTTP {resp.status_code}  ({len(resp.content)} bytes)")
    if resp.status_code != 200:
        print("  FAIL: non-200 response")
        return False

    try:
        root = ET.fromstring(resp.content)
    except ET.ParseError as e:
        print(f"  ERROR: XML parse failed: {e}")
        return False

    print(f"  Root tag     : {root.tag}")

    # Check namespaces
    ns_map = {k: v for k, v in root.attrib.items() if "xmlns" in k.lower()}
    if verbose:
        print(f"  Namespaces   : {ns_map}")

    # Find items
    _RDF_NS = "http://www.w3.org/1999/02/22-rdf-syntax-ns#"
    _DC_NS  = "http://purl.org/dc/elements/1.1/"
    els = root.findall(f"{{{_RDF_NS}}}item") or root.findall(f".//{{{_RDF_NS}}}item")
    bare_els = root.findall(".//item")
    print(f"  rdf:item     : {len(els)}")
    print(f"  bare <item>  : {len(bare_els)}")

    sample = els[0] if els else (bare_els[0] if bare_els else None)
    if sample is None:
        print("  FAIL: no <item> elements found")
        return False

    # Check expected fields on sample item
    checks = {
        "rdf:about attr"  : sample.get(f"{{{_RDF_NS}}}about"),
        "dc:title"        : sample.findtext(f"{{{_DC_NS}}}title"),
        "bare title"      : sample.findtext("title"),
        "dc:date"         : sample.findtext(f"{{{_DC_NS}}}date"),
        "bare pubDate"    : sample.findtext("pubDate"),
        "dc:description"  : sample.findtext(f"{{{_DC_NS}}}description"),
        "bare description": sample.findtext("description"),
        "bare link"       : sample.findtext("link"),
    }
    print("\n  Sample item field check:")
    all_ok = True
    for field, val in checks.items():
        status = "OK" if val else "MISSING"
        print(f"    {status:<8} {field}: {str(val)[:80] if val else 'None'}")
        if field in ("rdf:about attr", "dc:title", "bare title") and not val:
            all_ok = False

    # Validate that our parser would extract title + link
    title = (
        sample.findtext(f"{{{_DC_NS}}}title")
        or sample.findtext("title")
        or ""
    ).strip()
    link = (
        sample.get(f"{{{_RDF_NS}}}about")
        or sample.findtext("link")
        or ""
    ).strip()

    print(f"\n  Parser output (title): {title[:80]}")
    print(f"  Parser output (link) : {link[:80]}")

    if not title or not link:
        print("\n  RESULT: FAIL - parser cannot extract title/link from live feed")
        print("  ACTION: Update _load_tdnet_feed() parsing logic in news_ingester.py")
        return False

    print(f"\n  RESULT: PASS - TDnet parsing is compatible with live feed")
    return True


# ---------------------------------------------------------------------------
# J-Quants validation
# ---------------------------------------------------------------------------

def validate_jquants(refresh_token: str, symbol: str = "9432", verbose: bool = False) -> bool:
    print("\n[J-Quants] Exchanging refresh token for id_token...")
    try:
        resp = requests.get(
            f"{_JQUANTS_BASE}/token/auth_refresh",
            params={"refreshtoken": refresh_token},
            timeout=_TIMEOUT,
            headers={"User-Agent": _USER_AGENT},
        )
    except Exception as e:
        print(f"  ERROR: {e}")
        return False

    print(f"  HTTP {resp.status_code}")
    if resp.status_code != 200:
        print(f"  FAIL: {resp.text[:200]}")
        return False

    id_token = resp.json().get("idToken")
    if not id_token:
        print("  FAIL: idToken not found in response")
        print(f"  Response keys: {list(resp.json().keys())}")
        return False
    print(f"  id_token     : {id_token[:20]}...")

    print(f"\n[J-Quants] Fetching /fins/announcement for code={symbol}...")
    try:
        resp2 = requests.get(
            f"{_JQUANTS_BASE}/fins/announcement",
            params={"code": symbol},
            headers={"Authorization": f"Bearer {id_token}", "User-Agent": _USER_AGENT},
            timeout=_TIMEOUT,
        )
    except Exception as e:
        print(f"  ERROR: {e}")
        return False

    print(f"  HTTP {resp2.status_code}")
    if resp2.status_code != 200:
        print(f"  FAIL: {resp2.text[:200]}")
        return False

    try:
        data = resp2.json()
    except Exception as e:
        print(f"  ERROR: JSON parse failed: {e}")
        return False

    top_keys = set(data.keys())
    print(f"  Top-level keys: {top_keys}")

    # Check for 'announcement' wrapper
    if "announcement" not in data:
        print(f"  FAIL: 'announcement' key not found. Actual keys: {top_keys}")
        print("  ACTION: Update _fetch_jquants_announcements() to use correct key")
        return False

    entries = data["announcement"]
    print(f"  Entries count : {len(entries)}")

    if not entries:
        print("  WARN: no entries returned for this symbol")
        return True

    sample = entries[0]
    print(f"\n  Sample entry keys: {list(sample.keys())}")

    expected = {"CompanyName", "DocumentType", "Date", "URL"}
    missing = expected - set(sample.keys())
    if missing:
        print(f"\n  FAIL: missing expected keys: {missing}")
        print(f"  Actual keys: {list(sample.keys())}")
        print("  ACTION: Update _fetch_jquants_announcements() field names")
        return False

    if verbose:
        print(f"\n  Sample entry:")
        for k, v in sample.items():
            print(f"    {k}: {str(v)[:80]}")

    print(f"\n  RESULT: PASS - J-Quants parsing is compatible with live API")
    return True


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Validate TDnet and J-Quants live API responses against parser expectations."
    )
    ap.add_argument("--symbol", default="9432", help="4-digit TSE code to test (default: 9432)")
    ap.add_argument(
        "--jquants-token",
        dest="jquants_token",
        default=os.environ.get("JQUANTS_REFRESH_TOKEN", ""),
        help="J-Quants refresh token (or set JQUANTS_REFRESH_TOKEN env var)",
    )
    ap.add_argument("--verbose", action="store_true", help="Print full response details")
    args = ap.parse_args()

    results: dict[str, bool] = {}

    results["tdnet"] = validate_tdnet(verbose=args.verbose)

    if args.jquants_token:
        results["jquants"] = validate_jquants(
            args.jquants_token, symbol=args.symbol, verbose=args.verbose
        )
    else:
        print("\n[J-Quants] Skipped — set JQUANTS_REFRESH_TOKEN or pass --jquants-token")
        results["jquants"] = None  # type: ignore[assignment]

    print("\n" + "=" * 55)
    print("VALIDATION SUMMARY")
    print("=" * 55)
    for src, ok in results.items():
        if ok is None:
            print(f"  SKIP   {src}")
        elif ok:
            print(f"  PASS   {src}")
        else:
            print(f"  FAIL   {src}  <- update parser in news_ingester.py")
    print("=" * 55)

    failed = [k for k, v in results.items() if v is False]
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
