"""Tests for experiment_log preregistration module."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiment_log import (
    ALLOWED_CATEGORIES,
    count_trials,
    paradigm_shift_flag,
    preregister,
    record_outcome,
)


@pytest.fixture
def log_path(tmp_path: Path) -> Path:
    return tmp_path / "experiment_log.jsonl"


def _sample_window() -> dict:
    return {
        "train_start": "2021-01-01",
        "train_end": "2024-12-31",
        "validation_start": "2025-01-01",
        "validation_end": "2025-06-30",
    }


def test_preregister_writes_append_only(log_path: Path) -> None:
    eid = preregister(
        category="factor",
        hypothesis="roa_op has positive IC on 5d horizon",
        params={"horizon_days": 5, "winsorize": 0.01},
        data_window=_sample_window(),
        author="test",
        log_path=log_path,
    )
    assert eid.startswith("2") and "factor" in eid

    rows = [json.loads(l) for l in log_path.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 1
    assert rows[0]["status"] == "preregistered"
    assert rows[0]["param_hash"] == eid.split("__")[-1]


def test_record_outcome_appends_does_not_mutate(log_path: Path) -> None:
    eid = preregister(
        category="threshold",
        hypothesis="entry_score > 0.8 filter",
        params={"threshold": 0.8},
        data_window=_sample_window(),
        author="test",
        log_path=log_path,
    )
    record_outcome(
        eid,
        metrics={"ir": 0.42, "t_stat": 2.1, "n_obs": 1250},
        status="executed",
        log_path=log_path,
    )
    rows = [json.loads(l) for l in log_path.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 2
    assert rows[0]["status"] == "preregistered"
    assert rows[1]["status"] == "executed"
    assert rows[1]["metrics"]["ir"] == 0.42


def test_record_outcome_requires_existing_prereg(log_path: Path) -> None:
    with pytest.raises(LookupError):
        record_outcome(
            "nonexistent__factor__abc123",
            metrics={"ir": 0.1},
            log_path=log_path,
        )


def test_invalid_category_rejected(log_path: Path) -> None:
    with pytest.raises(ValueError):
        preregister(
            category="not_a_category",
            hypothesis="bogus",
            params={},
            data_window=_sample_window(),
            author="test",
            log_path=log_path,
        )


def test_data_window_requires_train_bounds(log_path: Path) -> None:
    with pytest.raises(ValueError):
        preregister(
            category="factor",
            hypothesis="test",
            params={"x": 1},
            data_window={"train_start": "2020-01-01"},
            author="test",
            log_path=log_path,
        )


def test_count_trials_filters(log_path: Path) -> None:
    for i in range(3):
        preregister(
            category="factor",
            hypothesis=f"factor_{i}",
            params={"i": i},
            data_window=_sample_window(),
            author="t",
            log_path=log_path,
        )
    preregister(
        category="threshold",
        hypothesis="th_1",
        params={"v": 1},
        data_window=_sample_window(),
        author="t",
        log_path=log_path,
    )
    assert count_trials(log_path=log_path) == 4
    assert count_trials(category="factor", log_path=log_path) == 3
    assert count_trials(category="threshold", log_path=log_path) == 1


def test_paradigm_shift_flag(log_path: Path) -> None:
    for i in range(4):
        preregister(
            category="factor",
            hypothesis=f"accruals_inv definition revision v{i}",
            params={"version": i},
            data_window=_sample_window(),
            author="t",
            log_path=log_path,
        )
    assert paradigm_shift_flag("accruals_inv", threshold=3, log_path=log_path) is True
    assert paradigm_shift_flag("roa_op", threshold=3, log_path=log_path) is False


def test_param_hash_deterministic(log_path: Path, tmp_path: Path) -> None:
    other = tmp_path / "other.jsonl"
    eid1 = preregister(
        category="factor",
        hypothesis="h",
        params={"a": 1, "b": 2},
        data_window=_sample_window(),
        author="t",
        log_path=log_path,
    )
    eid2 = preregister(
        category="factor",
        hypothesis="h",
        params={"b": 2, "a": 1},  # key order flipped
        data_window=_sample_window(),
        author="t",
        log_path=other,
    )
    assert eid1.split("__")[-1] == eid2.split("__")[-1]


def test_allowed_categories_nonempty() -> None:
    assert "factor" in ALLOWED_CATEGORIES
    assert "threshold" in ALLOWED_CATEGORIES
    assert len(ALLOWED_CATEGORIES) >= 5
