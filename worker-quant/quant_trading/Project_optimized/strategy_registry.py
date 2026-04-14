"""strategy_registry.py — single source of truth for real + paper strategies.

Each strategy entry binds a `strategy_id` (the row key used across
`orders`, `fills`, `positions`, `account_snapshots`) to:
  - Which signal module produces the target weights
  - Parameters (factor list, top_k, min_adv, etc.)
  - Capital tier: `real` (hits SBI) vs `paper` (shadow only)
  - Walk-forward evidence summary (last measured)
  - Kill-switch state: `active` | `paused` | `retired`

This file is the source everything else reads from — make_decision, the
paper runners, briefing, execution_gap_report. No strategy should exist
outside this registry.

2026-04-14 seed: post-walk-forward-verdict allocation.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
import json


@dataclass
class Evidence:
    """Last measured OOS evidence for a strategy.

    Not updated automatically yet — fill in from walk_forward_runner
    outputs manually. Date stamp lets us detect stale entries.
    """
    measured_at: str              # YYYY-MM-DD
    window: str                   # "2024-01 → 2026-04"
    n_periods: int
    cum_net: float                # e.g. 0.30 for +30%
    sharpe_ann: float
    maxdd: float
    monthly_excess_vs_ew: float   # +0.02 = +2% / month
    oos_holdout_result: str       # "positive" | "negative" | "untested"
    dsr_p: float | None = None    # Deflated Sharpe p-value vs trials


@dataclass
class StrategyEntry:
    strategy_id: str
    tier: str                     # "real" | "paper"
    signal_module: str            # e.g. "sprint_signal" | "alpha_factor_score"
    params: dict = field(default_factory=dict)
    capital_cap_jpy: float = 0.0
    state: str = "active"         # "active" | "paused" | "retired"
    last_paused_reason: str | None = None
    evidence: Evidence | None = None
    notes: str = ""


# ── Seed registry (2026-04-14 verdict) ────────────────────────────────
#
# Real-money strategies (must have evidence.monthly_excess_vs_ew > 0
# OR be an existing sunk position).
# Paper strategies have no capital cap — they just produce PnL rows for
# augmentation / execution-gap measurement.

_REGISTRY: dict[str, StrategyEntry] = {
    # ── sprint: existing real account, sunk 3041.T position ─────────
    "sprint": StrategyEntry(
        strategy_id="sprint",
        tier="real",
        signal_module="sprint_signal",
        params={"factors": ["mom_consist", "high52w"], "top_k": 3},
        capital_cap_jpy=250_000,  # Current 3041.T position ~¥234k — cap near it
        state="paused_sunk_only",
        last_paused_reason="walk-forward 2026-04-14: 2f composite −0.13%/mo vs EW; only hold existing sunk positions until exited",
        evidence=Evidence(
            measured_at="2026-04-14",
            window="2024-01 → 2026-04",
            n_periods=23,
            cum_net=0.261,
            sharpe_ann=0.85,
            maxdd=-0.137,
            monthly_excess_vs_ew=-0.0013,
            oos_holdout_result="untested",
            dsr_p=0.0,
        ),
        notes="vol_z removed 2026-04-14 (was negative alpha). Current 3041.T held; no new entries until evidence improves.",
    ),

    # ── high52w: new real strategy, most conservative candidate ──────
    "high52w": StrategyEntry(
        strategy_id="high52w",
        tier="real",
        signal_module="alpha_factor_score",
        params={"factors": ["high52w"], "top_k": 5, "min_adv": 50_000_000},
        capital_cap_jpy=100_000,
        state="active",
        evidence=Evidence(
            measured_at="2026-04-14",
            window="2024-01 → 2026-04",
            n_periods=23,
            cum_net=0.301,
            sharpe_ann=0.88,
            maxdd=-0.131,
            monthly_excess_vs_ew=0.0002,
            oos_holdout_result="untested",
            dsr_p=0.0,
        ),
        notes="Lowest-risk WF candidate. Matches EW return with comparable DD.",
    ),

    # ── amihud: aggressive real strategy, regime-sensitive ──────────
    "amihud": StrategyEntry(
        strategy_id="amihud",
        tier="real",
        signal_module="alpha_factor_score",
        params={"factors": ["alpha_amihud_20"], "top_k": 20, "min_adv": 100_000_000},
        capital_cap_jpy=100_000,
        state="active",
        evidence=Evidence(
            measured_at="2026-04-14",
            window="2024-01 → 2026-04",
            n_periods=23,
            cum_net=0.729,
            sharpe_ann=0.67,
            maxdd=-0.157,
            monthly_excess_vs_ew=0.0187,
            oos_holdout_result="negative_2023",  # ← important caveat
            dsr_p=0.0,
        ),
        notes="High IS edge but REGIME-DEPENDENT: 2023 OOS cum -3.1%. Kill-switch if 3m real PnL < -5%.",
    ),

    # ── Paper shadows of each real strategy ──────────────────────────
    "sprint_paper": StrategyEntry(
        strategy_id="sprint_paper",
        tier="paper",
        signal_module="sprint_signal",
        params={"factors": ["mom_consist", "high52w"], "top_k": 3},
        state="active",
        notes="Shadow of sprint. Used to measure execution gap real − paper.",
    ),
    "high52w_paper": StrategyEntry(
        strategy_id="high52w_paper",
        tier="paper",
        signal_module="alpha_factor_score",
        params={"factors": ["high52w"], "top_k": 5, "min_adv": 50_000_000},
        state="active",
    ),
    "amihud_paper": StrategyEntry(
        strategy_id="amihud_paper",
        tier="paper",
        signal_module="alpha_factor_score",
        params={"factors": ["alpha_amihud_20"], "top_k": 20, "min_adv": 100_000_000},
        state="active",
    ),

    # ── Paper-only experiments (parameter variants + bold bets) ──────
    "amihud_k5_paper": StrategyEntry(
        strategy_id="amihud_k5_paper",
        tier="paper",
        signal_module="alpha_factor_score",
        params={"factors": ["alpha_amihud_20"], "top_k": 5, "min_adv": 100_000_000},
        state="active",
        notes="Amihud with aggressive concentration — test k sensitivity.",
    ),
    "amihud_k30_paper": StrategyEntry(
        strategy_id="amihud_k30_paper",
        tier="paper",
        signal_module="alpha_factor_score",
        params={"factors": ["alpha_amihud_20"], "top_k": 30, "min_adv": 100_000_000},
        state="active",
    ),
    "amihud_adv50_paper": StrategyEntry(
        strategy_id="amihud_adv50_paper",
        tier="paper",
        signal_module="alpha_factor_score",
        params={"factors": ["alpha_amihud_20"], "top_k": 20, "min_adv": 50_000_000},
        state="active",
    ),
    "min_ret_paper": StrategyEntry(
        strategy_id="min_ret_paper",
        tier="paper",
        signal_module="alpha_factor_score",
        params={"factors": ["alpha_min_ret_20"], "top_k": 10},
        state="active",
        notes="Defensive — 20d min-return factor. MaxDD -3.3% in WF.",
    ),
    "mom_high52w_paper": StrategyEntry(
        strategy_id="mom_high52w_paper",
        tier="paper",
        signal_module="alpha_factor_score",
        params={"factors": ["mom_consist", "high52w"], "top_k": 5},
        state="active",
        notes="2-factor composite — is averaging better than high52w alone?",
    ),
}


# ── Accessors ─────────────────────────────────────────────────────────

def get(strategy_id: str) -> StrategyEntry | None:
    return _REGISTRY.get(strategy_id)


def list_all() -> list[StrategyEntry]:
    return list(_REGISTRY.values())


def list_real() -> list[StrategyEntry]:
    return [s for s in _REGISTRY.values() if s.tier == "real"]


def list_paper() -> list[StrategyEntry]:
    return [s for s in _REGISTRY.values() if s.tier == "paper"]


def list_active_real() -> list[StrategyEntry]:
    return [s for s in _REGISTRY.values()
            if s.tier == "real" and s.state == "active"]


# ── Serialization (for briefing / debug) ──────────────────────────────

def to_json(path: str | Path) -> None:
    Path(path).write_text(
        json.dumps(
            {k: _as_dict(v) for k, v in _REGISTRY.items()},
            indent=2, ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def _as_dict(entry: StrategyEntry) -> dict:
    d = asdict(entry)
    return d


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "dump":
        to_json("reports/strategy_registry.json")
        print("→ reports/strategy_registry.json")
    else:
        print(f"{len(_REGISTRY)} strategies ({len(list_real())} real, {len(list_paper())} paper)")
        for s in _REGISTRY.values():
            mark = "[R]" if s.tier == "real" else "[p]"
            cap = f"JPY{s.capital_cap_jpy:,.0f}" if s.capital_cap_jpy else "-"
            ev = f"exM={s.evidence.monthly_excess_vs_ew:+.4f}" if s.evidence else "-"
            print(f"  {mark} {s.strategy_id:<24} tier={s.tier:<5} cap={cap:<12} state={s.state:<18} {ev}")
