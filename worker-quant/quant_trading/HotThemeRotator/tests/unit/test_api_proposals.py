"""API contract tests for /api/proposals (Rule 13.18 L6 UI gate)."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
from fastapi.testclient import TestClient


def _write_proposal(base: Path, proposal_id: str, *, state: str = "proposals",
                    with_parameter_change: bool = False) -> dict:
    """Stage a proposal JSON on disk so the API can list/accept/reject."""
    payload = {
        "proposal_id": proposal_id,
        "created_ts": "2026-05-28T10:00:00+09:00",
        "snapshot_id": "snap_abcd1234",
        "trace_id": "trace_efgh5678",
        "evidence_class": "funnel_loss",
        "intervention_target": "chase_threshold_pct",
        "sample_size": 500,
        "confidence_interval": [0.10, 0.15],
        "counterfactual_validity": "exact_replay",
        "rationale_pointer": "reports/reflections/rca/2026-05-28.json",
        "generator": "structured_rca_v1",
        "parameter_change": (
            {"chase_threshold_pct": 8.0} if with_parameter_change else None
        ),
        "backtest_evidence": (
            {"pnl_improvement": 0.012, "n_periods": 30} if with_parameter_change else None
        ),
        "extra": {
            "source_layer": "L4_RCA",
            "source_trace_ids": ["trace_efgh5678"],
            "config_before_hash": "cfg_abc",
            "candidate_config_hash": "cfg_xyz",
            "outcome_window": "30d",
            "denominator_counts": {"eligible": 100, "scored": 80},
        },
    }
    # Match decision_gate.proposal_dir() layout: {base}/reports/reflections/{state}
    dir_path = base / "reports" / "reflections" / state
    dir_path.mkdir(parents=True, exist_ok=True)
    (dir_path / f"{proposal_id}.json").write_text(
        json.dumps(payload), encoding="utf-8",
    )
    return payload


@pytest.fixture
def client(monkeypatch):
    base = Path(".runtime") / "proposals_api_tests"
    if base.exists():
        shutil.rmtree(base, ignore_errors=True)
    base.mkdir(parents=True, exist_ok=True)

    import api.proposals as prop_mod
    monkeypatch.setattr(prop_mod, "PROJECT_ROOT", base)

    from api.main import create_app
    app = create_app()
    yield TestClient(app), base
    shutil.rmtree(base, ignore_errors=True)


def test_list_empty(client):
    c, _ = client
    resp = c.get("/api/proposals")
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["counts"] == {"proposals": 0, "accepted": 0, "rejected": 0, "expired": 0}
    assert payload["items"] == []


def test_list_pending_after_stage(client):
    c, base = client
    _write_proposal(base, "abc123")
    payload = c.get("/api/proposals").json()
    assert payload["counts"]["proposals"] == 1
    assert payload["items"][0]["proposal_id"] == "abc123"


def test_list_specific_state(client):
    c, base = client
    _write_proposal(base, "x1", state="proposals")
    _write_proposal(base, "x2", state="accepted")
    payload = c.get("/api/proposals?state=accepted").json()
    assert payload["state"] == "accepted"
    assert len(payload["items"]) == 1
    assert payload["items"][0]["proposal_id"] == "x2"


def test_list_invalid_state_returns_422(client):
    c, _ = client
    resp = c.get("/api/proposals?state=invalid_state")
    assert resp.status_code == 422


def test_items_carry_rule_13_6_metadata(client):
    """Rule 13.18.1 — full Rule 13.6 metadata must be exposed by API."""
    c, base = client
    _write_proposal(base, "meta1")
    item = c.get("/api/proposals").json()["items"][0]
    for k in (
        "proposal_id", "created_ts", "snapshot_id", "trace_id",
        "evidence_class", "intervention_target", "sample_size",
        "confidence_interval", "counterfactual_validity",
        "rationale_pointer", "generator",
    ):
        assert k in item


def test_items_annotate_expiry_fields(client):
    """Rule 13.18.4 — expiry banner needs age_days + is_expired flags."""
    c, base = client
    _write_proposal(base, "exp1")
    item = c.get("/api/proposals").json()["items"][0]
    assert "age_days" in item
    assert "is_expired_by_age" in item
    assert "expiry_days_remaining" in item


def test_items_flag_shadow_disclosure_when_parameter_change(client):
    """Rule 13.18.2 — items with parameter_change must surface requires_shadow_disclosure=True."""
    c, base = client
    _write_proposal(base, "p1", with_parameter_change=True)
    _write_proposal(base, "p2", with_parameter_change=False)
    items = c.get("/api/proposals").json()["items"]
    by_id = {i["proposal_id"]: i for i in items}
    assert by_id["p1"]["requires_shadow_disclosure"] is True
    assert by_id["p2"]["requires_shadow_disclosure"] is False


# ─── Accept ──────────────────────────────────────────────────────────────


def test_accept_unknown_proposal_returns_404(client):
    c, _ = client
    resp = c.post("/api/proposals/nonexistent/accept", json={})
    assert resp.status_code == 404


def test_accept_parameter_change_requires_shadow_confirm(client):
    """Rule 13.18.2 — parameter_change accept without user_confirm_shadow → 400."""
    c, base = client
    _write_proposal(base, "pparamshadow", with_parameter_change=True)
    resp = c.post("/api/proposals/pparamshadow/accept", json={"user_confirm_shadow": False})
    assert resp.status_code == 400
    assert "shadow_disclosure_required" in resp.text


def test_accept_non_parameter_change_no_shadow_required(client):
    """Diagnostic-only proposals can be accepted without shadow disclosure.
    B1 fix — was previously asserting 200|409 to tolerate 500; now must be 200.
    """
    c, base = client
    _write_proposal(base, "pdiag", with_parameter_change=False)
    resp = c.post("/api/proposals/pdiag/accept", json={"user_confirm_shadow": False})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["ok"] is True
    assert body["new_state"] == "accepted"
    # Verify the file actually moved
    assert (base / "reports" / "reflections" / "accepted" / "pdiag.json").exists()
    assert not (base / "reports" / "reflections" / "proposals" / "pdiag.json").exists()


def test_accept_parameter_change_with_shadow_confirm_returns_200(client):
    """B1 happy-path with shadow_confirm — was hidden by 200|409 tolerance."""
    c, base = client
    _write_proposal(base, "pparam", with_parameter_change=True)
    resp = c.post("/api/proposals/pparam/accept", json={"user_confirm_shadow": True})
    assert resp.status_code == 200, resp.text
    assert resp.json()["new_state"] == "accepted"


# ─── Reject ──────────────────────────────────────────────────────────────


def test_reject_with_invalid_reason_returns_422(client):
    c, base = client
    _write_proposal(base, "r1")
    resp = c.post("/api/proposals/r1/reject", json={"reason": "not_in_allow_list"})
    assert resp.status_code == 422
    assert "invalid_reject_reason" in resp.text


def test_reject_with_allowed_reason_returns_200(client):
    """B1 — must succeed end-to-end (was hidden by 200|409 tolerance)."""
    c, base = client
    _write_proposal(base, "r2happy")
    resp = c.post("/api/proposals/r2happy/reject", json={"reason": "user_disagrees"})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["ok"] is True
    assert body["new_state"] == "rejected"
    assert body["reason"] == "user_disagrees"
    # Verify file moved
    assert (base / "reports" / "reflections" / "rejected" / "r2happy.json").exists()
    assert not (base / "reports" / "reflections" / "proposals" / "r2happy.json").exists()


def test_reject_unknown_proposal_returns_404(client):
    c, _ = client
    resp = c.post("/api/proposals/nonexistent/reject", json={"reason": "other"})
    assert resp.status_code == 404


# ─── Rule 11.5 whitelist — only the documented POST paths exist ──────────


def test_no_batch_accept_endpoint(client):
    """Rule 13.18.5 — no 'accept all' / 'reject all' batch endpoint exists.

    /api/proposals/accept_all matches the {proposal_id} catch-all but the path
    must include /accept or /reject suffix, so it returns 405 (method not
    allowed on /api/proposals/accept_all root) or 404. We accept either as
    proof there's no actual batch handler — the important property is that
    no batch operation succeeds.
    """
    c, _ = client
    accept_all = c.post("/api/proposals/accept_all", json={})
    reject_all = c.post("/api/proposals/reject_all", json={})
    assert accept_all.status_code in (404, 405)
    assert reject_all.status_code in (404, 405)
