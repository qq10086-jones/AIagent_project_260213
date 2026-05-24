"""Tests for theme_heat_adapter (P8-12 / ADR-0005)."""
import sqlite3
import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from hot_theme_rotator.data.theme_heat_adapter import (  # noqa: E402
    ThemeHeatAdapterError,
    ThemeHeatRow,
    default_db_path,
    load_theme_heat,
)


def _create_db(tmp_path: Path, rows: list[tuple]) -> Path:
    db = tmp_path / "japan_market.db"
    conn = sqlite3.connect(db)
    conn.execute("""
        CREATE TABLE factor_signals (
            asof TEXT, symbol TEXT, factor_name TEXT,
            raw_score REAL, z_score REAL,
            created_at TEXT, pred_return REAL
        )
    """)
    conn.executemany(
        "INSERT INTO factor_signals VALUES (?, ?, ?, ?, ?, ?, ?)", rows
    )
    conn.commit(); conn.close()
    return db


# Synthetic but realistic: 3 factors × 4 symbols at one asof.
# mom_20 has the extreme z's, sharpe_20 medium, value_bp mild.
_FIXTURE = [
    # mom_20: extreme z's → high heat
    ("2026-04-13", "8035.T", "mom_20", 0.4, 2.5, "ts", None),
    ("2026-04-13", "6920.T", "mom_20", 0.3, 2.1, "ts", None),
    ("2026-04-13", "7203.T", "mom_20", -0.2, -1.8, "ts", None),
    ("2026-04-13", "1306.T", "mom_20", 0.1, 0.5, "ts", None),
    # sharpe_20: medium z's
    ("2026-04-13", "8035.T", "sharpe_20", 0.5, 1.0, "ts", None),
    ("2026-04-13", "6920.T", "sharpe_20", 0.4, 0.8, "ts", None),
    ("2026-04-13", "7203.T", "sharpe_20", -0.1, -0.5, "ts", None),
    # value_bp: small z's → low heat
    ("2026-04-13", "8035.T", "value_bp", 0.1, 0.2, "ts", None),
    ("2026-04-13", "7203.T", "value_bp", 0.05, 0.1, "ts", None),
    # Older asof (should be ignored — we use MAX(asof) only)
    ("2026-04-12", "8035.T", "mom_20", 0.4, 3.0, "ts", None),
]


def test_load_returns_top_themes_ranked_by_aggregate_strength(tmp_path):
    db = _create_db(tmp_path, _FIXTURE)
    themes = load_theme_heat(db, top_n=3)
    assert len(themes) == 3
    # mom_20 has highest mean |z|, then sharpe_20, then value_bp
    assert themes[0].id == "mom_20"
    assert themes[1].id == "sharpe_20"
    assert themes[2].id == "value_bp"


def test_heat_scales_with_mean_abs_z_score(tmp_path):
    db = _create_db(tmp_path, _FIXTURE)
    themes = load_theme_heat(db, top_n=3)
    mom = themes[0]
    # mom_20 mean|z| = (2.5 + 2.1 + 1.8 + 0.5) / 4 = 1.725 → heat ≈ 86
    assert 80 <= mom.heat <= 90
    sharpe = themes[1]
    # sharpe mean|z| = (1.0 + 0.8 + 0.5) / 3 ≈ 0.77 → heat ≈ 38
    assert 35 <= sharpe.heat <= 45


def test_leaders_are_top_abs_z_symbols_per_theme(tmp_path):
    db = _create_db(tmp_path, _FIXTURE)
    themes = load_theme_heat(db, top_n=3, leaders_per_theme=3)
    mom = themes[0]
    assert mom.leaders == ("8035.T", "6920.T", "7203.T")  # by abs(z): 2.5, 2.1, 1.8


def test_labels_use_chinese_when_known_factor(tmp_path):
    db = _create_db(tmp_path, _FIXTURE)
    themes = load_theme_heat(db, top_n=3)
    mom = themes[0]
    assert mom.label == "20 日动量"  # from _FACTOR_LABELS


def test_label_falls_back_to_factor_id_when_unknown(tmp_path):
    rows = [
        ("2026-04-13", "1.T", "made_up_factor", 0.1, 1.0, "ts", None),
        ("2026-04-13", "2.T", "made_up_factor", 0.1, 1.5, "ts", None),
    ]
    db = _create_db(tmp_path, rows)
    themes = load_theme_heat(db, top_n=1)
    assert themes[0].label == "made_up_factor"


def test_older_asof_rows_ignored(tmp_path):
    db = _create_db(tmp_path, _FIXTURE)
    themes = load_theme_heat(db, top_n=3)
    mom = themes[0]
    # 2026-04-12 has 8035.T mom_20 z=3.0 but should be ignored since we use MAX(asof)=2026-04-13
    # So heat is computed only over 2026-04-13 rows
    assert mom.asof == "2026-04-13"


def test_null_z_score_rows_are_skipped(tmp_path):
    rows = _FIXTURE + [
        ("2026-04-13", "9999.T", "mom_20", 0.0, None, "ts", None),  # null z — skip
    ]
    db = _create_db(tmp_path, rows)
    themes = load_theme_heat(db, top_n=3)
    # mom_20 leaders still 8035/6920/7203 (9999 excluded)
    assert "9999.T" not in themes[0].leaders


def test_top_n_caps_returned_themes(tmp_path):
    db = _create_db(tmp_path, _FIXTURE)
    themes = load_theme_heat(db, top_n=2)
    assert len(themes) == 2


def test_empty_factor_signals_returns_empty(tmp_path):
    db = _create_db(tmp_path, [])
    assert load_theme_heat(db) == ()


def test_fails_closed_on_missing_table(tmp_path):
    db = tmp_path / "japan_market.db"
    conn = sqlite3.connect(db); conn.execute("CREATE TABLE other(x INT)")
    conn.commit(); conn.close()
    with pytest.raises(ThemeHeatAdapterError, match="factor_signals"):
        load_theme_heat(db)


def test_fails_closed_on_missing_db(tmp_path):
    with pytest.raises(ThemeHeatAdapterError, match="not found"):
        load_theme_heat(tmp_path / "nope.db")


def test_default_db_path():
    assert default_db_path().name == "japan_market.db"
