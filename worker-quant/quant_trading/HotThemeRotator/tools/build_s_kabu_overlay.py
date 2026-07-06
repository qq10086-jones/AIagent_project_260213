"""Emit the S株 universe overlay snapshot (Rule 5.2 / task #5).

Writes ``reports/screener/s_kabu_overlay_{asof}.json`` — the held + watchlisted
names that S株 unlocks (lot-untradable but S株-tradable), for the candidate panel
to merge alongside the sibling screener short list. Read-only against the sibling
screener (ADR-0005); it only reads HTR-owned journal + user_state + price DB.

Usage:
  python tools/build_s_kabu_overlay.py --asof 2026-06-24
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path


def _ensure_src_on_path() -> None:
    src = Path(__file__).resolve().parent.parent / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))


def main() -> int:
    _ensure_src_on_path()
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    from hot_theme_rotator.candidate_engine.s_kabu_universe import build_s_kabu_overlay

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--asof", default=date.today().isoformat())
    ap.add_argument("--base-dir", default=".")
    ap.add_argument("--account-jpy", type=float, default=400_000.0)
    args = ap.parse_args()

    ov = build_s_kabu_overlay(args.base_dir, account_jpy=args.account_jpy)
    ov["asof"] = args.asof
    out = Path(args.base_dir) / "reports" / "screener" / f"s_kabu_overlay_{args.asof}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(ov, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"wrote {out}: {len(ov['candidates'])} S株 candidate(s) "
        f"from {len(ov['names_considered'])} held/watchlist name(s)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
