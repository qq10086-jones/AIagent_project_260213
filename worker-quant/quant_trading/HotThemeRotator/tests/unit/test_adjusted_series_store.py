"""P35-03 tests — shared adjusted-series loader over a raw price DB."""
import sqlite3
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from hot_theme_rotator.data.adjusted_series_store import (  # noqa: E402
    load_adjusted_series,
    window_is_clean,
)


def _db(tmp_path, rows):
    db = tmp_path / "prices.db"
    conn = sqlite3.connect(str(db))
    conn.execute("create table daily_prices(symbol text, date text, open real, "
                 "high real, low real, close real, volume real)")
    conn.executemany("insert into daily_prices values (?,?,?,?,?,?,?)",
                     [(s, d, c, c, c, c, v) for s, d, c, v in rows])
    conn.commit()
    conn.close()
    return db


def test_split_with_volume_is_adjusted_like_1306t(tmp_path):
    db = _db(tmp_path, [
        ("1306.T", "2026-03-27", 3800.0, 1_000_000),
        ("1306.T", "2026-03-28", 3817.0, 1_100_000),
        ("1306.T", "2026-03-30", 376.4, 9_800_000),
        ("1306.T", "2026-03-31", 380.1, 9_500_000),
    ])
    s = load_adjusted_series(db)["1306.T"]
    assert s.error is None and s.ambiguous == []
    rets = [s.closes[i + 1] / s.closes[i] - 1 for i in range(3)]
    assert all(abs(r) < 0.05 for r in rets), "the -90.1% artifact must be gone"
    assert s.closes[-1] == 380.1


def test_ambiguous_jump_is_reported_per_window(tmp_path):
    db = _db(tmp_path, [                      # -50% with NO volume -> ambiguous
        ("X.T", "2026-03-01", 100.0, None),
        ("X.T", "2026-03-02", 50.0, None),
        ("X.T", "2026-03-03", 51.0, None),
        ("X.T", "2026-03-04", 52.0, None),
    ])
    s = load_adjusted_series(db)["X.T"]
    assert s.ambiguous == [1]
    assert not window_is_clean(s, 0, 2)       # crosses the jump
    assert window_is_clean(s, 2, 3)           # after the jump: computable


def test_clean_symbol_and_symbol_filter(tmp_path):
    db = _db(tmp_path, [("A.T", "2026-03-01", 100.0, 1e6),
                        ("A.T", "2026-03-02", 101.0, 1e6),
                        ("B.T", "2026-03-01", 200.0, 1e6)])
    only_a = load_adjusted_series(db, symbols=["A.T"])
    assert set(only_a) == {"A.T"}
    assert window_is_clean(only_a["A.T"], 0, 1)


def test_invalid_series_surfaces_error_not_silence(tmp_path):
    db = _db(tmp_path, [("BAD.T", "2026-03-02", 100.0, 1e6),
                        ("BAD.T", "2026-03-02", 101.0, 1e6)])  # duplicate date
    s = load_adjusted_series(db)["BAD.T"]
    assert s.error is not None and "duplicate" in s.error
    assert not window_is_clean(s, 0, 0)
