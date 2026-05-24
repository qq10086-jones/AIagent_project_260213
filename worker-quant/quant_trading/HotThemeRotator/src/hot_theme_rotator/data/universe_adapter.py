"""Read-only adapter for Project_optimized's screener output (ADR-0005).

Reads `Project_optimized/selected_tickers.json` — the daily top-N short list
produced by the screener. Each entry has rich features (alpha_score, mom_20,
mom_60, sharpe_20, adv_rank, vol_adj_mom20, vol_z, close, ...). This adapter
returns them as structured rows so `api.serializers` can build real
`OpportunityCandidate` rows instead of relying on `build_sample_panel`.

Strictly read-only. No writes to Project_optimized.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


# Field families pulled from selected_tickers.json. The full row carries many
# more — these are the ones the dashboard consumes.
_REQUIRED_TOP_KEYS = ("asof", "symbols", "details")
_REQUIRED_DETAIL_KEYS = ("symbol", "score", "close", "adv", "vol")


class UniverseAdapterError(RuntimeError):
    """Raised when selected_tickers.json cannot be safely read."""


@dataclass(frozen=True)
class ScreenedTicker:
    """One row from the daily screener short list."""

    symbol: str
    score: float                # 0..1 raw alpha score from screener
    close: float                # screener's recorded close at asof
    adv: float                  # average dollar volume
    vol: float                  # realized volatility
    hard_fail: bool
    reason: str
    mom_20: float = 0.0
    mom_60: float = 0.0
    sharpe_20: float = 0.0
    adv_rank: float = 0.0
    fundamental_score: float = 0.0
    raw: dict = None


@dataclass(frozen=True)
class ScreenerSnapshot:
    """The daily top-N short list with metadata."""

    asof: str
    version: str
    count: int
    tickers: tuple[ScreenedTicker, ...]
    source_path: str = ""


def load_screener_snapshot(path: str | Path) -> ScreenerSnapshot:
    """Load the daily screener output (selected_tickers.json).

    Fail-closed on missing file / malformed JSON / missing required keys.
    """
    src = Path(path)
    if not src.exists():
        raise UniverseAdapterError(f"selected_tickers.json not found: {src}")
    try:
        payload = json.loads(src.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise UniverseAdapterError(
            f"selected_tickers.json is not valid JSON: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise UniverseAdapterError("selected_tickers.json root must be an object")
    missing_top = [k for k in _REQUIRED_TOP_KEYS if k not in payload]
    if missing_top:
        raise UniverseAdapterError(
            f"selected_tickers.json missing required keys: {missing_top}"
        )

    details = payload.get("details") or []
    if not isinstance(details, list):
        raise UniverseAdapterError("`details` must be a list")

    tickers: list[ScreenedTicker] = []
    for row in details:
        if not isinstance(row, dict):
            continue
        missing = [k for k in _REQUIRED_DETAIL_KEYS if k not in row]
        if missing:
            raise UniverseAdapterError(
                f"ticker row missing required keys {missing}: {row.get('symbol', '?')}"
            )
        tickers.append(
            ScreenedTicker(
                symbol=str(row["symbol"]),
                score=float(row["score"]),
                close=float(row["close"]),
                adv=float(row["adv"]),
                vol=float(row["vol"]),
                hard_fail=bool(row.get("hard_fail", False)),
                reason=str(row.get("reason", "")),
                mom_20=float(row.get("mom_20", 0.0) or 0.0),
                mom_60=float(row.get("mom_60", 0.0) or 0.0),
                sharpe_20=float(row.get("sharpe_20", 0.0) or 0.0),
                adv_rank=float(row.get("adv_rank", 0.0) or 0.0),
                fundamental_score=float(row.get("fundamental_score", 0.0) or 0.0),
                raw=dict(row),
            )
        )

    return ScreenerSnapshot(
        asof=str(payload["asof"]),
        version=str(payload.get("version", "")),
        count=int(payload.get("count", len(tickers))),
        tickers=tuple(tickers),
        source_path=str(src),
    )


def default_selected_tickers_path(project_optimized_root: str | Path | None = None) -> Path:
    """Default location of `selected_tickers.json`."""
    if project_optimized_root is not None:
        return Path(project_optimized_root) / "selected_tickers.json"
    here = Path(__file__).resolve()
    return here.parents[4] / "Project_optimized" / "selected_tickers.json"
