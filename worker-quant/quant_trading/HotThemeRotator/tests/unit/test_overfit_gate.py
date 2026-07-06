"""Anti-overfit promotion gate (ADR-0010 P17-3)."""
from __future__ import annotations

from hot_theme_rotator.calibration.overfit_gate import (
    deflated_sharpe_ratio,
    expected_max_sharpe,
    promote_gate,
)


def test_expected_max_sharpe_grows_with_trials():
    assert expected_max_sharpe(1, 0.1) == 0.0          # one trial → no inflation
    e10 = expected_max_sharpe(10, 0.05)
    e1000 = expected_max_sharpe(1000, 0.05)
    assert 0 < e10 < e1000                              # more search → higher noise ceiling


def test_deflated_sharpe_falls_as_trials_rise():
    # Same observed Sharpe looks great after 1 trial, worthless after thousands.
    high = deflated_sharpe_ratio(0.15, n_trials=1, n_obs=250, sr_std=0.05)
    low = deflated_sharpe_ratio(0.15, n_trials=5000, n_obs=250, sr_std=0.05)
    assert high > 0.95 > low


def test_deflated_sharpe_rises_with_sample_size():
    short = deflated_sharpe_ratio(0.12, n_trials=10, n_obs=40, sr_std=0.05)
    long = deflated_sharpe_ratio(0.12, n_trials=10, n_obs=750, sr_std=0.05)
    assert long > short


def test_promote_gate_passes_strong_few_trials():
    v = promote_gate(0.15, n_trials=1, n_obs=250, sr_std=0.05)
    assert v["pass"] is True and v["reasons"] == []
    assert v["dsr"] >= 0.95


def test_promote_gate_fails_noise_after_many_trials():
    # A Sharpe at/below the best-of-N noise outcome must not promote.
    v = promote_gate(0.15, n_trials=5000, n_obs=250, sr_std=0.05)
    assert v["pass"] is False
    assert any("DSR" in r for r in v["reasons"])
    assert v["expectedMaxSharpe"] > 0.15               # noise ceiling exceeds the observed SR


def test_promote_gate_fails_small_sample():
    v = promote_gate(0.30, n_trials=1, n_obs=20, sr_std=0.05)
    assert v["pass"] is False and any("sample" in r for r in v["reasons"])


def test_promote_gate_requires_declared_trial_count():
    v = promote_gate(0.20, n_trials=0, n_obs=250, sr_std=0.05)
    assert v["pass"] is False and any("trial count" in r for r in v["reasons"])


def test_promote_gate_rejects_nonpositive_sr_std():
    # Codex fix: sr_std<=0 must fail closed (can't deflate without cross-trial dispersion).
    v = promote_gate(0.20, n_trials=10, n_obs=250, sr_std=0.0)
    assert v["pass"] is False and any("sr_std" in r for r in v["reasons"])
