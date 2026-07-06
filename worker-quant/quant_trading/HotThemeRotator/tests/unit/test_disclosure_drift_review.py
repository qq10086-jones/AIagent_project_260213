"""Disclosure-drift review harness (ADR-0010 P17-4)."""
from __future__ import annotations

from hot_theme_rotator.reporting.disclosure_drift_review import build_disclosure_drift_review


def _ev(ticker, title, price, ts="2026-06-17T15:00:00+09:00"):
    return {"ticker": ticker, "title": title, "price": price, "published_ts": ts}


def test_empty_corpus_is_insufficient_data():
    r = build_disclosure_drift_review([], event_return_fn=lambda e, h: 0.02)
    assert r["verdict"] == "insufficient_data"
    assert r["actionableCount"] == 0 and r["eventsIn"] == 0


def test_exclusion_ledger_accounts_for_every_drop():
    events = [
        _ev("1.T", "通期業績予想の上方修正", 900),                       # actionable (¥900 tradable)
        _ev("2.T", "通期業績予想の上方修正", 900, ts=""),                # no PIT timestamp
        _ev("3.T", "本社移転に関するお知らせ", 900),                     # not material
        _ev("4.T", "配当に関するお知らせ", 900),                         # material(dividend) but no direction
        _ev("5.T", "通期業績予想の上方修正", 72760),                     # material+dir but untradable (¥72k)
    ]
    r = build_disclosure_drift_review(events, event_return_fn=lambda e, h: 0.02, min_events=1)
    assert r["excluded"]["not_pit"] == 1
    assert r["excluded"]["not_material"] == 1
    assert r["excluded"]["no_direction"] == 1
    assert r["excluded"]["not_tradable"] == 1
    assert r["actionableCount"] == 1
    # every event is accounted for: actionable + all exclusions == eventsIn
    assert r["actionableCount"] + sum(r["excluded"].values()) == r["eventsIn"]


def test_net_drift_is_in_signal_direction_and_cost_adjusted():
    # Upward-revision event, +3% raw move → positive net (after ~30-40bps cost).
    events = [_ev(f"{i}.T", "通期業績予想の上方修正", 900) for i in range(5)]
    r = build_disclosure_drift_review(events, event_return_fn=lambda e, h: 0.03, min_events=1)
    h3 = r["horizons"]["3"]
    assert h3["n"] == 5 and h3["meanNet"] > 0 and h3["meanNet"] < 0.03   # positive but cost reduces it
    assert h3["hitRate"] == 1.0


def test_downward_event_with_drop_is_positive_drift_in_direction():
    # Downward-revision (direction -1) that then FALLS -3% → drift in predicted dir is +.
    events = [_ev(f"{i}.T", "通期業績予想の下方修正", 900) for i in range(5)]
    r = build_disclosure_drift_review(events, event_return_fn=lambda e, h: -0.03, min_events=1)
    assert r["horizons"]["3"]["meanNet"] > 0   # (-0.03)*(-1) - cost > 0


def test_promotion_is_deferred_to_trial_matrix():
    # DSR/PBO is NOT auto-run in the per-event review (Codex fix): it needs the true
    # cross-trial Sharpe dispersion from a real trial matrix, run separately.
    events = [_ev(f"{i}.T", "通期業績予想の上方修正", 900) for i in range(30)]
    r = build_disclosure_drift_review(events, event_return_fn=lambda e, h: 0.02, min_events=20)
    assert r["verdict"] == "evaluated"
    assert r["promotion"]["eligible"] is True
    assert "promote" not in r["horizons"]["3"]                      # no muddled scalar DSR here
    assert "overfit_gate.promote_gate" in r["promotion"]["note"]


def test_unmatured_returns_are_ledgered_and_gate_the_verdict():
    # The slowest horizon (10d) is unmatured → counted in nMissing, and the verdict is
    # gated insufficient even though 3d/5d have data (Codex fix: no silent drop, slowest-horizon gate).
    events = [_ev(f"{i}.T", "通期業績予想の上方修正", 900) for i in range(5)]
    r = build_disclosure_drift_review(events, event_return_fn=lambda e, h: (None if h == 10 else 0.02), min_events=1)
    assert r["horizons"]["10"]["n"] == 0 and r["horizons"]["10"]["nMissing"] == 5
    assert r["horizons"]["3"]["n"] == 5
    assert r["verdict"] == "insufficient_data"
