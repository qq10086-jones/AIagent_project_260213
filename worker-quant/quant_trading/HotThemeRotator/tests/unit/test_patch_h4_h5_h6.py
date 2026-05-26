"""Patch tests for H4 (journal lock) / H5 (correction reference) / H6 (API 409)."""
from __future__ import annotations

import os
import sys
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from fastapi.testclient import TestClient  # noqa: E402

from api.main import create_app  # noqa: E402
from hot_theme_rotator.portfolio.journal_writer import (  # noqa: E402
    PortfolioJournalError,
    append_cash_event,
    append_fill,
    journal_lock,
    read_all_journal,
)
from hot_theme_rotator.portfolio.manual_entry_service import (  # noqa: E402
    build_cash_event,
    build_fill_entry,
    commit_fill,
    preview_fill,
)
from hot_theme_rotator.portfolio.schema import (  # noqa: E402
    FillEntry,
    derive_fill_entry_id,
)
from hot_theme_rotator.portfolio.validation import PortfolioValidationError  # noqa: E402


JST = timezone(timedelta(hours=9), name="JST")
NOW = datetime(2026, 5, 26, 10, 0, tzinfo=JST)


def _seed_path_a(tmp_path):
    deposit = build_cash_event(ts="2026-05-07T09:00:00+09:00", amount=389345.0,
                               reason="deposit", note="initial")
    buy = build_fill_entry(side="BUY", symbol="1306.T", qty=900, price=403.0,
                           ts="2026-05-07T09:30:00+09:00")
    append_cash_event(deposit, base_dir=tmp_path)
    append_fill(buy, base_dir=tmp_path)


# ─── H4: journal_lock serialization ─────────────────────────────────────────


def test_h4_lock_acquire_release(tmp_path):
    with journal_lock(tmp_path):
        lock_file = tmp_path / "reports" / "portfolio" / ".journal.lock"
        assert lock_file.exists()
    assert not lock_file.exists()


def test_h4_lock_blocks_second_acquirer_until_timeout(tmp_path):
    with journal_lock(tmp_path, timeout=0.5):
        # While held, another acquirer with short timeout must raise.
        with pytest.raises(PortfolioJournalError, match="lock"):
            with journal_lock(tmp_path, timeout=0.1):
                pass


def test_h4_lock_released_after_normal_exit(tmp_path):
    with journal_lock(tmp_path, timeout=0.5):
        pass
    # Should immediately re-acquire.
    with journal_lock(tmp_path, timeout=0.5):
        pass


def test_h4_lock_released_after_exception(tmp_path):
    with pytest.raises(RuntimeError):
        with journal_lock(tmp_path, timeout=0.5):
            raise RuntimeError("boom")
    # Lock must still be released.
    with journal_lock(tmp_path, timeout=0.5):
        pass


def test_h4_concurrent_commits_serialize(tmp_path):
    """Two threads attempting commit_fill must serialize via the lock; the
    second one's re-validation should reject the now-impossible second SELL."""
    _seed_path_a(tmp_path)

    fill_a = build_fill_entry(side="SELL", symbol="1306.T", qty=900, price=420.0,
                              ts="2026-05-25T14:00:00+09:00", note="A")
    fill_b = build_fill_entry(side="SELL", symbol="1306.T", qty=900, price=420.0,
                              ts="2026-05-25T14:00:00+09:00", note="B")
    preview_a = preview_fill(fill_a, base_dir=tmp_path, now=NOW)
    preview_b = preview_fill(fill_b, base_dir=tmp_path, now=NOW)

    results = {"a": None, "b": None}

    def commit_a():
        try:
            results["a"] = commit_fill(preview_a, base_dir=tmp_path, now=NOW)
        except Exception as exc:
            results["a"] = exc

    def commit_b():
        # Stagger slightly so A grabs lock first deterministically.
        time.sleep(0.02)
        try:
            results["b"] = commit_fill(preview_b, base_dir=tmp_path, now=NOW)
        except Exception as exc:
            results["b"] = exc

    t1 = threading.Thread(target=commit_a)
    t2 = threading.Thread(target=commit_b)
    t1.start(); t2.start(); t1.join(); t2.join()

    # Exactly one wins; the loser sees oversell-via-revalidation.
    successes = [r for r in results.values() if isinstance(r, Path)]
    failures = [r for r in results.values() if isinstance(r, PortfolioValidationError)]
    assert len(successes) == 1
    assert len(failures) == 1


# ─── H5: correction reference validation ────────────────────────────────────


def _build_correction(*, ts, corrects, side="BUY", qty=100, price=400.0,
                     symbol="1306.T", note="reverses prior"):
    eid = derive_fill_entry_id(
        ts=ts, symbol=symbol, side=side, qty=qty, price=price,
        source="correction", note=note,
    )
    return FillEntry(
        entry_id=eid, ts=ts, symbol=symbol, side=side, qty=qty, price=price,
        source="correction", fee=0.0, note=note, corrects=corrects,
    )


def test_h5_correction_referencing_nonexistent_entry_rejected(tmp_path):
    _seed_path_a(tmp_path)
    bogus_target = "0000000000000000"
    correction = _build_correction(ts="2026-05-25T14:00:00+09:00", corrects=bogus_target)
    with pytest.raises(PortfolioJournalError, match="missing prior entry_id"):
        append_fill(correction, base_dir=tmp_path)


def test_h5_correction_referencing_existing_entry_accepted(tmp_path):
    _seed_path_a(tmp_path)
    existing = read_all_journal(tmp_path)
    target = next(e for e in existing if isinstance(e, FillEntry))
    correction = _build_correction(
        ts="2026-05-25T14:00:00+09:00", corrects=target.entry_id,
    )
    path = append_fill(correction, base_dir=tmp_path)
    assert path.exists()


def test_h5_commit_rejects_fill_that_is_pre_corrected(tmp_path):
    """Attacker pre-stages a correction targeting the deterministic id of a fill
    the user is about to commit. Commit must refuse, not silently let the fill
    land and then be dropped by skip-both."""
    _seed_path_a(tmp_path)
    # Compute the entry_id of the about-to-be-committed legitimate SELL.
    legitimate = build_fill_entry(
        side="SELL", symbol="1306.T", qty=400, price=417.6,
        ts="2026-05-25T14:00:00+09:00",
    )
    # We can't actually append a correction targeting it (writer-level check
    # would reject), so simulate the attack by first appending the target via
    # a different code path, then a correction, then trying to commit the
    # original — except the writer would also reject the dup. The realistic
    # forward-targeting attack requires a correction created with the id
    # known in advance; we test the commit-time guard directly.
    #
    # First: append a real fill to serve as a corrected target.
    a = build_fill_entry(side="SELL", symbol="1306.T", qty=100, price=420.0,
                        ts="2026-05-25T11:00:00+09:00", note="real fill A")
    append_fill(a, base_dir=tmp_path)
    # Now append a correction targeting a *future* legitimate fill — but the
    # writer guard prevents this. So we craft the correction to target an
    # existing fill (A), then for the test, manually inject a synthesized
    # journal scenario: the correction targets `legitimate.entry_id` and is
    # already in journal. Easiest: bypass writer by direct file write.
    corr = _build_correction(
        ts="2026-05-25T11:30:00+09:00", corrects=legitimate.entry_id,
    )
    # Direct-file append to bypass writer guard (simulates a malicious actor).
    import json
    journal_file = tmp_path / "reports" / "portfolio" / "journal" / "2026-05-25.jsonl"
    journal_file.parent.mkdir(parents=True, exist_ok=True)
    payload = {"_type": "fill", **corr.to_dict()}
    with journal_file.open("a", encoding="utf-8") as h:
        h.write(json.dumps(payload, ensure_ascii=False, sort_keys=True))
        h.write("\n")

    preview = preview_fill(legitimate, base_dir=tmp_path, now=NOW)
    with pytest.raises(PortfolioJournalError, match="already targets it"):
        commit_fill(preview, base_dir=tmp_path, now=NOW)


# ─── H6: API 409 on commit re-validation failure ────────────────────────────


@pytest.fixture
def client(tmp_path, monkeypatch):
    # Redirect the API's PROJECT_ROOT to a tmp dir so we don't write the real one.
    import api.portfolio_fill as pf
    monkeypatch.setattr(pf, "PROJECT_ROOT", tmp_path)
    return TestClient(create_app())


def test_h6_post_fill_returns_409_on_commit_revalidation_failure(client, tmp_path):
    """Set up a journal where a SELL would oversell, then POST commit=True.
    Expected: 409 conflict with refresh-preview detail."""
    # Empty journal — no holdings — a SELL preview path will hard-reject in
    # validation; that's a 400. To trigger commit-time re-validation failure,
    # we need preview to pass but commit's re-validation to fail. Easiest:
    # seed enough cash, preview a BUY which passes, then between preview &
    # commit we manipulate journal... but TestClient is synchronous.
    #
    # Substitute scenario: forward-target attack. Inject a correction
    # targeting a known entry_id, then POST commit=True for that exact fill.
    import json
    from hot_theme_rotator.portfolio.schema import derive_fill_entry_id

    proposed_id = derive_fill_entry_id(
        ts="2026-05-25T14:00:00+09:00", symbol="1306.T", side="BUY", qty=1,
        price=400.0, source="manual", note="",
    )
    # First: a real fill so the correction has a corrects target.
    real = build_fill_entry(side="BUY", symbol="1306.T", qty=10, price=400.0,
                           ts="2026-05-25T09:00:00+09:00")
    append_fill(real, base_dir=tmp_path)
    # Then: inject correction-of-future-id via direct file write.
    corr = _build_correction(
        ts="2026-05-25T09:30:00+09:00", corrects=proposed_id,
    )
    journal_file = tmp_path / "reports" / "portfolio" / "journal" / "2026-05-25.jsonl"
    payload = {"_type": "fill", **corr.to_dict()}
    with journal_file.open("a", encoding="utf-8") as h:
        h.write(json.dumps(payload, ensure_ascii=False, sort_keys=True))
        h.write("\n")

    # Need cash so preview validation passes — seed a deposit too.
    deposit = build_cash_event(ts="2026-05-25T08:00:00+09:00", amount=10000.0,
                               reason="deposit")
    append_cash_event(deposit, base_dir=tmp_path)

    resp = client.post("/api/portfolio/fill", json={
        "side": "BUY", "symbol": "1306.T", "qty": 1, "price": 400.0,
        "ts": "2026-05-25T14:00:00+09:00", "commit": True,
    })
    assert resp.status_code == 409
    assert "re-validation" in resp.json()["detail"].lower() or "targets" in resp.json()["detail"].lower()


def test_h6_post_fill_returns_400_on_preview_validation_failure(client, tmp_path):
    """A 400 path still works — bad schema or hard-gate at preview time."""
    resp = client.post("/api/portfolio/fill", json={
        "side": "SELL", "symbol": "1306.T", "qty": 10, "price": 400.0,
        "ts": "2027-01-01T10:00:00+09:00", "commit": True,
    })
    # Future ts → validation 400 (preview-time)
    assert resp.status_code == 400
