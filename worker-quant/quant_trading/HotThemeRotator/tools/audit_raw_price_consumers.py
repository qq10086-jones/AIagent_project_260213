"""P35-01b — inventory every consumer of raw ``daily_prices`` that computes returns.

    python tools/audit_raw_price_consumers.py --asof 2026-08-08

Per the adopted post-P34 priority #1: the split defect is not fixed by patching
one tool — it is fixed by knowing exactly which code paths turn raw prices into
returns, and migrating each to the ``adjusted_prices`` contract (or documenting
why raw is correct there, e.g. ADV/turnover).

Writes `reports/research/raw_price_consumers_{asof}.json`. Static text scan —
same honesty terms as the P34-00 audit: it finds references, not executions.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import date
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

_RETURNY = re.compile(
    r"pct_change|/ *prev|prev *[*/]|- *1\.0|- *1\b.*close|close.*/.*close|"
    r"ret(urn)?s?\b|nxt */ *prev|b */ *a *- *1", re.IGNORECASE)
_ADOPTED = re.compile(r"adjusted_prices|adjusted_returns|detect_price_jumps|"
                      r"detect_corporate_actions|longest_clean_segment")


def main(argv: list[str] | None = None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--asof", default=date.today().isoformat())
    ap.add_argument("--base-dir", default=str(PROJECT_ROOT))
    args = ap.parse_args(argv)

    base = Path(args.base_dir).resolve()
    consumers = []
    for root in ("src", "tools", "api"):
        for p in sorted((base / root).rglob("*.py")):
            rel_parts = p.relative_to(base).parts
            if "__pycache__" in rel_parts or ".runtime" in rel_parts:
                continue
            try:
                text = p.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue
            if "daily_prices" not in text:
                continue
            rel = str(p.relative_to(base)).replace("\\", "/")
            consumers.append({
                "file": rel,
                "computes_returns": bool(_RETURNY.search(text)),
                "uses_adjusted_contract": bool(_ADOPTED.search(text)),
                "status": (
                    "guarded_or_migrated" if _ADOPTED.search(text)
                    else "SUSPECT_returns_on_raw" if _RETURNY.search(text)
                    else "raw_ok_no_returns"),
            })

    suspect = [c for c in consumers if c["status"] == "SUSPECT_returns_on_raw"]
    payload = {
        "_kind": "raw_price_consumer_inventory",
        "asof": args.asof,
        "generated_by": "tools/audit_raw_price_consumers.py",
        "n_consumers": len(consumers),
        "n_suspect": len(suspect),
        "n_guarded_or_migrated": sum(
            1 for c in consumers if c["status"] == "guarded_or_migrated"),
        "consumers": consumers,
        "limits": [
            "static text scan: finds references, not executions",
            "'computes_returns' is a regex heuristic; each SUSPECT entry needs a "
            "human read before migration",
        ],
        "governance": {"task": "P35-01b", "rules": ["Rule 3 advice-only"]},
    }
    out = base / "reports" / "research" / f"raw_price_consumers_{args.asof}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"daily_prices consumers: {len(consumers)}")
    for c in consumers:
        print(f"  [{c['status']:<24}] {c['file']}")
    print(f"\nSUSPECT (returns on raw, unguarded): {len(suspect)}")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
