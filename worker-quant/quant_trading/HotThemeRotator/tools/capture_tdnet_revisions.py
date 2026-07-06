"""P23-A capture runner — rescue + daily capture of TDnet revision documents.

TDnet's public site serves documents for only ~31 days (verified 2026-07-06:
a 2024 URL is 404). This tool scans the local TDnet metadata corpus for
guidance-revision rows within the still-alive window, downloads their PDFs,
parses the A/B revision magnitudes, and stores both (see
``tdnet_revision_docs``). Run daily (after the TDnet poll) and the corpus of
REAL revision magnitudes accumulates forward. Research-only.

Usage:
    python tools\\capture_tdnet_revisions.py [--days 35]
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hot_theme_rotator.data.external.tdnet_revision_docs import (  # noqa: E402
    capture_revisions,
)


def _reparse() -> int:
    """Re-run the parser over already-stored PDFs (no network) and rebuild the
    revisions jsonl files in place — used after parser improvements."""
    import json as _json

    from hot_theme_rotator.data.external.tdnet_revision_docs import (
        extract_text,
        parse_revision_text,
    )

    rev_dir = ROOT / "reports" / "tdnet_docs" / "revisions"
    pdf_dir = ROOT / "reports" / "tdnet_docs" / "pdf"
    total = parsed_n = 0
    for f in sorted(rev_dir.glob("*.jsonl")):
        out_rows = []
        for line in f.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rec = _json.loads(line)
            pdf = pdf_dir / f"{rec['doc_id']}.pdf"
            if pdf.exists():
                new = parse_revision_text(extract_text(pdf.read_bytes()))
                rec["parsed"] = new["parsed"]
                rec["metrics"] = new["metrics"]
            out_rows.append(rec)
            total += 1
            parsed_n += 1 if rec.get("parsed") else 0
        with f.open("w", encoding="utf-8") as fh:
            for rec in out_rows:
                fh.write(_json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"[reparse] {parsed_n}/{total} parsed")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=35)
    ap.add_argument("--throttle", type=float, default=1.0)
    ap.add_argument("--reparse", action="store_true",
                    help="re-parse stored PDFs only (no network)")
    args = ap.parse_args()

    if args.reparse:
        return _reparse()

    cutoff = (_dt.date.today() - _dt.timedelta(days=args.days)).isoformat()
    rows = []
    seen_files = set()
    # scan BOTH corpora: the production daily poll (reports/tdnet, since
    # 2026-07-01) and the 2-year probe backfill (.runtime/tdnet_probe) whose
    # recent files are still inside TDnet's ~31-day document window.
    for corpus in (ROOT / "reports" / "tdnet",
                   ROOT / ".runtime" / "tdnet_probe" / "reports" / "tdnet"):
        if not corpus.exists():
            continue
        for f in sorted(corpus.glob("*.jsonl")):
            if f.stem < cutoff or f.stem in seen_files:
                continue
            seen_files.add(f.stem)
            for line in f.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    rows.append(json.loads(line))
    print(f"[capture] corpus rows within {args.days}d window: {len(rows)}")
    stats = capture_revisions(ROOT, rows, throttle_seconds=args.throttle)
    print(f"[capture] {stats}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
