"""TDnet guidance-revision document capture + magnitude parser (P23-A, Lane A).

Feasibility probe 2026-07-06 established two facts that define this module:
(1) revision PDFs parse cleanly (前回発表予想Ａ / 今回修正予想Ｂ rows extract
with pdfplumber); (2) TDnet's public site serves documents for only ~31 days —
the two-year metadata corpus's old URLs are 404. Revision DOCUMENTS are
therefore perishable data: this lane captures them FORWARD (daily, plus a
~30-day rescue window) and derives real revision magnitudes — superseding the
failed title-regex surprise (P19-04 diagnosis: title carries direction in only
51/749 cases).

Storage (separate research lane, never touches the §10 decision log):
- reports/tdnet_docs/pdf/{docid}.pdf              raw document (audit)
- reports/tdnet_docs/revisions/{date}.jsonl       parsed magnitudes per event

Research-only: magnitudes are descriptive facts; no probability, no advice.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Optional

METRIC_COLUMNS = (
    # aliases matched on the SPACE-NORMALIZED header line (PDFs often letter-space
    # Japanese: 売 上 高); first-index order on that line defines column order.
    # Bank/insurer/REIT top-lines added 2026-07-06 (correctness review finding 2:
    # an unrecognized leading column shifted every metric one place, fabricating
    # a surprise on the wrong line).
    ("revenue", ("売上高", "営業収益", "売上収益", "経常収益", "営業総収入",
                 "保険料等収入", "営業収入")),
    ("operating_income", ("営業利益",)),
    ("ordinary_income", ("経常利益", "税引前利益", "経常損益")),
    ("net_income", ("純利益",)),  # 当期純利益 / 親会社株主に帰属する当期純利益
)

_ROW_A = "前回発表予想"
_ROW_B_RE = re.compile(r"今回(修正|発表)予想")  # both wordings occur in the wild

_NUM_RE = re.compile(r"^[△▲\-−]?\d{1,3}(?:,\d{3})*(?:\.\d+)?$")


def _parse_number(token: str) -> Optional[float]:
    t = token.strip()
    if t in ("未定", "―", "—", "－", "-", ""):
        return None
    neg = t[0] in "△▲-−"
    if neg:
        t = t[1:]
    if not _NUM_RE.match(t):
        return None
    try:
        v = float(t.replace(",", ""))
    except ValueError:
        return None
    return -v if neg else v


def _row_numbers(line: str, n_cols: int) -> list[Optional[float]]:
    """Pull the first ``n_cols`` numeric-or-未定 tokens after the row label."""
    tail = re.split(r"[（(][ＡＢAB][－-]?[ＡＢAB]?[）)]", line, maxsplit=1)
    body = tail[1] if len(tail) > 1 else line
    out: list[Optional[float]] = []
    for tok in body.split():
        if "円" in tok:  # per-share yen figures end the numeric block
            break
        v = _parse_number(tok)
        if v is None and tok.strip() not in ("未定", "―", "—", "－", "-"):
            continue
        out.append(v)
        if len(out) >= n_cols:
            break
    while len(out) < n_cols:
        out.append(None)
    return out


def _norm(line: str) -> str:
    """Space-normalized view for LABEL/HEADER detection (letter-spaced PDFs)."""
    return line.replace(" ", "").replace("　", "")


# A header line is prose (not a column header) when it carries sentence
# connectors / predicates — such a line names every metric but is not the
# table header (correctness review finding 3).
_PROSE_MARKERS = ("及び", "並びに", "について", "予想を", "修正いた", "見通し",
                  "または", "反映", "。")
# Lines that carry numbers but are NOT a forecast row (finding 7 — a
# （参考）前期実績 actuals line must never be adopted as the prior forecast).
_NONFORECAST_MARKERS = ("前期実績", "参考", "増減額", "増減率", "実績値", "差異")


def _is_prose(norm_line: str) -> bool:
    return any(m in norm_line for m in _PROSE_MARKERS)


def _header_cols(norm_line: str) -> list[str]:
    """Metric columns named on a normalized header line, in appearance order."""
    found: list[tuple[int, str]] = []
    for name, aliases in METRIC_COLUMNS:
        idxs = [norm_line.find(a) for a in aliases if a in norm_line]
        if idxs:
            found.append((min(i for i in idxs if i >= 0), name))
    found.sort()
    return [name for _, name in found]


def _find_header(lines: list[str], a_idx: int) -> list[str]:
    """The table header for the A-row at ``a_idx``: the NEAREST non-prose,
    column-bearing line above it, merged with an immediately-adjacent
    continuation line (split header, e.g. '…に帰属する' / '当期純利益').
    Prose sentences that merely mention the metrics are skipped (finding 3)."""
    block: list[tuple[int, list[str]]] = []
    for back in range(a_idx - 1, max(-1, a_idx - 9), -1):
        nl = _norm(lines[back])
        if _is_prose(nl):
            if block:
                break  # header block ended
            continue
        cols = _header_cols(nl)
        if cols:
            block.append((back, cols))
        elif block:
            break  # a blank/non-header line closes the block
    if not block:
        return []
    # merge top→bottom (upper line's columns first), dedup preserving order
    merged: list[str] = []
    for _, cols in sorted(block):
        for c in cols:
            if c not in merged:
                merged.append(c)
    return merged


def _row_with_carryover(lines: list[str], i: int, n_cols: int) -> list[Optional[float]]:
    """Numbers for the labeled row at ``lines[i]``; when the label sits alone on
    its line (values wrapped), fall back to the NEXT line's numeric block — but
    NEVER adopt an A/B label line or a non-forecast reference line (前期実績 /
    増減 / 参考), which would fabricate a revision (findings 7)."""
    vals = _row_numbers(lines[i], n_cols)
    if all(v is None for v in vals) and i + 1 < len(lines):
        nxt = lines[i + 1]
        nn = _norm(nxt)
        if (_ROW_A not in nn and not _ROW_B_RE.search(nn)
                and not any(m in nn for m in _NONFORECAST_MARKERS)):
            carco = _row_numbers(nxt, n_cols)
            if any(v is not None for v in carco):
                return carco
    return vals


def parse_revision_text(text: str) -> dict[str, Any]:
    """Parse the first A/B forecast-revision block out of extracted PDF text.

    Returns ``{"parsed": bool, "metrics": {name: {before, after, pct}}}``.
    ``pct`` is computed only when both sides are numeric and A != 0 —
    未定 (undetermined) yields ``pct=None``, never a fabricated magnitude.
    Labels/headers are matched on space-normalized lines (letter-spaced PDFs);
    values wrapped onto the following line are picked up there.
    """
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    header_cols: list[str] = []
    row_a: Optional[list[Optional[float]]] = None
    row_b: Optional[list[Optional[float]]] = None
    for i, ln in enumerate(lines):
        nl = _norm(ln)
        if _ROW_A in nl and row_a is None:
            header_cols = _find_header(lines, i)  # nearest non-prose header block
            if not header_cols:
                continue
            row_a = _row_with_carryover(lines, i, len(header_cols))
        elif _ROW_B_RE.search(nl) and row_a is not None and row_b is None:
            row_b = _row_with_carryover(lines, i, len(header_cols))
            break
    if row_a is None or row_b is None or not header_cols:
        return {"parsed": False, "metrics": {}}
    metrics: dict[str, Any] = {}
    for j, name in enumerate(header_cols):
        a, b = row_a[j], row_b[j]
        # (b−a)/|a| — identical to b/a−1 for positive bases, but keeps the SIGN
        # honest when the prior guidance was a loss (a widening loss must read
        # as a NEGATIVE surprise, not +68%).
        pct = ((b - a) / abs(a)) if (a not in (None, 0) and b is not None) else None
        metrics[name] = {"before": a, "after": b, "pct": pct}
    return {"parsed": True, "metrics": metrics}


# ---------------------------------------------------------------------------
# capture (network + storage) — kept separate from the pure parser above
# ---------------------------------------------------------------------------

_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/126.0 Safari/537.36",
    "Accept": "application/pdf,*/*",
    "Referer": "https://www.release.tdnet.info/",
}


def _direct_url(url: str) -> str:
    return url.split("rd.php?")[-1] if "rd.php?" in url else url


def fetch_pdf(url: str, *, timeout: int = 30) -> Optional[bytes]:
    import requests

    r = requests.get(_direct_url(url), headers=_HEADERS, timeout=timeout)
    if r.status_code != 200 or r.content[:4] != b"%PDF":
        return None
    return r.content


def extract_text(pdf_bytes: bytes, *, max_pages: int = 3) -> str:
    import io

    import pdfplumber

    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        return "\n".join((p.extract_text() or "") for p in pdf.pages[:max_pages])


def is_revision_title(title: str) -> bool:
    return "業績予想" in title and "修正" in title


def capture_revisions(
    base_dir: Path | str,
    rows: list[dict[str, Any]],
    *,
    throttle_seconds: float = 1.0,
) -> dict[str, int]:
    """Fetch+parse+store revision documents for corpus ``rows`` (each with
    ticker/title/url/published_ts). Idempotent by doc id; per-doc failures are
    counted, never fatal."""
    import time

    base = Path(base_dir)
    pdf_dir = base / "reports" / "tdnet_docs" / "pdf"
    rev_dir = base / "reports" / "tdnet_docs" / "revisions"
    pdf_dir.mkdir(parents=True, exist_ok=True)
    rev_dir.mkdir(parents=True, exist_ok=True)

    stats = {"considered": 0, "fetched": 0, "parsed": 0, "skipped_existing": 0,
             "failed": 0}
    for row in rows:
        title = row.get("title", "")
        if not is_revision_title(title):
            continue
        stats["considered"] += 1
        url = _direct_url(row.get("url", ""))
        doc_id = url.rsplit("/", 1)[-1].replace(".pdf", "")
        if not doc_id:
            stats["failed"] += 1
            continue
        pdf_path = pdf_dir / f"{doc_id}.pdf"
        date = str(row.get("published_ts", ""))[:10] or "unknown"
        out_path = rev_dir / f"{date}.jsonl"
        if pdf_path.exists():
            stats["skipped_existing"] += 1
            continue
        try:
            blob = fetch_pdf(url)
            if blob is None:
                stats["failed"] += 1
                continue
            pdf_path.write_bytes(blob)
        except Exception:  # noqa: BLE001 — a fetch failure leaves nothing on disk
            stats["failed"] += 1
            time.sleep(max(throttle_seconds, 0.0))
            continue
        # Parse defensively: finding 4 — a parse crash must still leave a
        # jsonl record (parsed:false) so the on-disk PDF is recoverable via
        # --reparse; the previous code wrote the PDF then dropped the doc.
        try:
            parsed = parse_revision_text(extract_text(blob))
        except Exception as exc:  # noqa: BLE001
            parsed = {"parsed": False, "metrics": {}, "parse_error": str(exc)[:200]}
        record = {
            "doc_id": doc_id,
            "ticker": row.get("ticker"),
            "title": title,
            "published_ts": row.get("published_ts"),
            "url": url,
            **parsed,
        }
        with out_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        stats["fetched"] += 1
        if parsed.get("parsed"):
            stats["parsed"] += 1
        time.sleep(max(throttle_seconds, 0.0))
    return stats
