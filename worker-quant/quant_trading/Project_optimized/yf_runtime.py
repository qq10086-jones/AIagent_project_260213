from __future__ import annotations

from pathlib import Path

import yfinance as yf


def configure_yfinance_cache(base_dir: str | None = None) -> Path:
    cache_dir = Path(base_dir) if base_dir else (Path(__file__).resolve().parent / ".yf_cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    yf.set_tz_cache_location(str(cache_dir))
    return cache_dir
