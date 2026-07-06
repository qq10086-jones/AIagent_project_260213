"""Tests for the SKHY forward-evidence review harness (P20-07 / Rule 11.15, 16).

Default verdict is insufficient_data; promotion is NEVER granted by this harness
(it requires the ADR-0010 / Rule 16 gates). Live-only — backdated clusters are
excluded, never pooled. No probability/win-rate/expected-return field.
"""
from hot_theme_rotator.reporting.skhy_event_review import review_skhy_events


def _cluster(date, n=6, *, live=True, aligned=True):
    samples = []
    for i in range(n):
        symp = i / 10.0
        rel = (i / 10.0) if aligned else (-(i / 10.0))  # sympathy vs fwd-rel return
        samples.append({"symbol": f"{i}.T", "sympathy": symp, "fwd_rel_return": rel})
    return {"date": date, "live": live, "samples": samples}


def test_insufficient_data_below_cluster_floor():
    out = review_skhy_events([_cluster("2026-06-25"), _cluster("2026-06-26"), _cluster("2026-06-27")],
                             min_event_clusters=20)
    assert out["verdict"] == "insufficient_data"
    assert out["eventClusters"] == 3
    assert out["promotionAllowed"] is False
    assert out["rankIc5d"] is None


def test_dedup_same_day_clusters():
    # two clusters on the same date count as ONE event cluster
    out = review_skhy_events([_cluster("2026-06-25"), _cluster("2026-06-25")], min_event_clusters=20)
    assert out["eventClusters"] == 1


def test_excludes_backdated_clusters():
    clusters = [_cluster("2026-06-25", live=True), _cluster("2026-06-26", live=False)]
    out = review_skhy_events(clusters, min_event_clusters=20)
    assert out["eventClusters"] == 1  # backdated 06-26 excluded


def test_rank_ic_computed_when_enough_clusters_but_no_auto_promotion():
    clusters = [_cluster(f"2026-06-{d:02d}", aligned=True) for d in range(1, 26)]  # 25 distinct live days
    out = review_skhy_events(clusters, min_event_clusters=20)
    assert out["eventClusters"] == 25
    assert out["rankIc5d"] is not None
    assert out["rankIc5d"] > 0.0  # aligned sympathy↔return → positive IC
    assert out["promotionAllowed"] is False  # still needs ADR-0010 / Rule 16


def test_cost_hurdle_present_and_no_forbidden_fields():
    out = review_skhy_events([_cluster("2026-06-25")], min_event_clusters=20, cost_hurdle=0.04)
    assert out["costHurdle"] == 0.04
    for bad in ("probability", "win_rate", "winRate", "expected_return", "expectedReturn", "edge"):
        assert bad not in out


def test_notes_declare_live_only_and_gate_requirement():
    out = review_skhy_events([_cluster("2026-06-25")], min_event_clusters=20)
    joined = " ".join(out["notes"]).lower()
    assert "live-only" in joined
    assert "rule 16" in joined or "adr-0010" in joined


def test_missing_live_flag_is_not_counted_as_live_evidence():
    # P20 fix#2: a cluster WITHOUT an explicit live=True must NOT count
    no_flag = {"date": "2026-06-25", "samples": [{"sympathy": 0.1, "fwd_rel_return": 0.1}]}
    assert review_skhy_events([no_flag], min_event_clusters=20)["eventClusters"] == 0
    assert review_skhy_events([dict(no_flag, live=False)], min_event_clusters=20)["eventClusters"] == 0
    assert review_skhy_events([dict(no_flag, live=True)], min_event_clusters=20)["eventClusters"] == 1
