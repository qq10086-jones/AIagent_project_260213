from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd


@dataclass
class HumanReport:
    headline: str
    highlights: list[str]
    warnings: list[str]
    actions: list[str]


def _safe_float(val) -> float:
    try:
        return float(val)
    except Exception:
        return 0.0


def _base_symbol(symbol: str) -> str:
    s = (symbol or "").strip()
    if s.endswith(".T"):
        return s[:-2]
    return s


def find_symbol_variants(symbols: Iterable[str]) -> dict[str, set[str]]:
    variants: dict[str, set[str]] = {}
    for s in symbols:
        if not s:
            continue
        base = _base_symbol(s)
        variants.setdefault(base, set()).add(s)
    return variants


def detect_symbol_inconsistencies(
    orders: pd.DataFrame, fills: pd.DataFrame, positions: pd.DataFrame
) -> list[str]:
    all_syms = []
    for df, col in [(orders, "symbol"), (fills, "symbol"), (positions, "symbol")]:
        if col in df.columns and len(df) > 0:
            all_syms.extend(df[col].dropna().astype(str).tolist())
    variants = find_symbol_variants(all_syms)
    issues = []
    for base, syms in variants.items():
        if len(syms) > 1:
            issues.append(f"Symbol variants detected for {base}: {sorted(syms)}")
    return issues


def notional_mismatch_ratio(orders: pd.DataFrame, fills: pd.DataFrame) -> float:
    if len(orders) == 0 or len(fills) == 0:
        return 0.0
    o = orders.copy()
    if "expected_value" not in o.columns:
        return 0.0
    expected = float(o["expected_value"].fillna(0.0).sum())
    f = fills.copy()
    if "qty" not in f.columns or "price" not in f.columns:
        return 0.0
    f["notional"] = f["qty"] * f["price"]
    filled = float(f["notional"].fillna(0.0).sum())
    if expected <= 0 or filled <= 0:
        return 0.0
    hi = max(expected, filled)
    lo = min(expected, filled)
    return hi / lo if lo > 0 else 0.0


def load_target_weights(reports_dir: Path) -> pd.DataFrame:
    p = reports_dir / "target_weights.csv"
    if not p.exists():
        return pd.DataFrame()
    return pd.read_csv(p)


def load_weights_history(reports_dir: Path) -> pd.DataFrame:
    p = reports_dir / "weights_history.csv"
    if not p.exists():
        return pd.DataFrame()
    return pd.read_csv(p)


def build_human_report(
    asof: str,
    run_row: dict,
    orders: pd.DataFrame,
    fills: pd.DataFrame,
    positions: pd.DataFrame,
    account_snapshots: pd.DataFrame,
    target_weights: pd.DataFrame,
    weights_history: pd.DataFrame,
) -> HumanReport:
    highlights: list[str] = []
    warnings: list[str] = []
    actions: list[str] = []

    status = run_row.get("status", "")
    ts = run_row.get("ts", "")
    headline = f"Run {run_row.get('run_id','')} ({asof}) status={status} ts={ts}"

    # Concentration
    if len(target_weights) > 0 and "target_weight" in target_weights.columns:
        tw = target_weights.copy()
        tw["target_weight"] = tw["target_weight"].fillna(0.0).astype(float)
        top = tw.sort_values("target_weight", ascending=False).head(3)
        top_items = [f"{r['symbol']} {r['target_weight']*100:.1f}%" for _, r in top.iterrows()]
        highlights.append("Top weights: " + ", ".join(top_items))
        hh = float((top["target_weight"] ** 2).sum())
        if hh > 0.5:
            warnings.append("Portfolio is highly concentrated (top-3 weights dominate).")

    # Orders vs fills notional mismatch
    ratio = notional_mismatch_ratio(orders, fills)
    if ratio >= 5.0:
        warnings.append(f"Orders vs fills notional mismatch: ~{ratio:.1f}x.")
        actions.append("Check price/qty scale and symbol normalization in fills.")

    # Symbol inconsistencies
    sym_issues = detect_symbol_inconsistencies(orders, fills, positions)
    warnings.extend(sym_issues)
    if sym_issues:
        actions.append("Normalize symbols (e.g., unify 9432 vs 9432.T).")

    # Missing prices
    if len(positions) > 0 and "market_price" in positions.columns:
        missing = positions[positions["market_price"].isna()]
        if len(missing) > 0:
            warnings.append(f"Missing market_price for {len(missing)} positions.")
            actions.append("Update market DB or fix symbol mapping to prices.")

    # Account snapshot health
    if len(account_snapshots) > 0:
        latest = account_snapshots.sort_values("asof").iloc[-1]
        nav = _safe_float(latest.get("nav"))
        cash = _safe_float(latest.get("cash"))
        highlights.append(f"Latest NAV: {nav:,.0f}")
        if nav < 0 or cash < 0:
            warnings.append("Negative NAV or cash detected in account snapshots.")
            actions.append("Verify cashflow and fill ingestion; check for scale errors.")
    else:
        warnings.append("No account snapshots found.")
        actions.append("Run build_account_snapshot.py after importing fills.")

    # Weights history recency
    if len(weights_history) > 0 and "date" in weights_history.columns:
        latest_date = weights_history["date"].iloc[-1]
        highlights.append(f"Latest weight date: {latest_date}")

    if not highlights:
        highlights.append("No summary metrics available yet.")
    if not actions:
        actions.append("No immediate action required.")

    return HumanReport(headline=headline, highlights=highlights, warnings=warnings, actions=actions)

