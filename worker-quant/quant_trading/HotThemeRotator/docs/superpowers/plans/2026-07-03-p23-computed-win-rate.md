# P23 Computed Win-Rate Program — Implementation Plan (2026-07-03)

Owner directive: strategies with HIGH win rate where the win rate is **calculated, not guessed**.
The calculation machinery exists (isotonic calibrator → K-fold/walk-forward validators → DSR
promote gate → gated UI `calibrated_probability` display). Missing = predictive INPUTS.
Program = four data lanes; one shared acceptance test; kill criteria are what make a displayed
win rate "based on calculation".

## Shared acceptance test (every lane, no exceptions)

1. Rule 16.0 break-even derivation first (IC > τ·c_rt/σ_r at the intended horizon).
2. Historical event study / cross-sectional Rank-IC through the existing harness
   (PIT next-open entry, 5D/20D excess vs 1306.T, Rule 5.1 tradability, cost model).
3. `overfit_gate.promote_gate`: DSR ≥ 0.95, n_obs_effective ≥ 60, honest trial counting.
4. PASS → isotonic calibration → Rule 8.2.2/9.4.1 flip → UI shows calibrated probability.
   FAIL → lane closed permanently, logged in PROJECT_STATUS.
5. Win-rate geometry arithmetic wired to the Action Board later: P_be = risk/(risk+reward)
   + costs, displayed next to the setup's historical hit rate (n, CI).

## Asset audit (verified 2026-07-03)

- `EDINET_API_KEY`: in User env, validated live (200; 909 docs listed 2026-06-30).
- Project_v5 `src/data/edinet_loader.py`: working list client (transport-injectable).
  **Bug found**: v5 backfill script filtered `130` as annual; loader (correct) says
  `120=有価証券報告書`, `130=訂正`. Likely cause of the historical "too few rows".
  v5 `fundamental_snapshots` schema is PIT-ready but has 0 rows — the XBRL→values
  parser was never built. That parser is THE missing 20%.
- Local TDnet corpus: 21,959 disclosures (254 files), 749 業績予想修正, 377 配当予想修正;
  only 51/749 revision titles carry direction → magnitudes must come from documents.
- Tailscale up (100.118.51.81); remote mode live (Rule 15.9).

## Lane B — EDINET fundamental panel (BUILD NOW, free)

Goal: kill the "only 2–3 valid cross-sections" blocker → real value/quality factor test.

- New HTR-side module `src/hot_theme_rotator/data/external/edinet_fundamentals.py`
  (self-contained per ADR-0005 — no cross-tree import; patterns borrowed from v5 loader):
  - `EdinetClient`: `list_documents(date)` + `fetch_document(doc_id, doc_format)`
    (format 5 = CSV zip preferred; 1 = XBRL zip fallback), transport-injectable.
  - `parse_financial_csv(zip_bytes)`: EDINET type=5 CSV (UTF-16LE TSV inside zip);
    whitelist element extraction — NetSales / OperatingIncome / OrdinaryIncome /
    Profit(Loss)AttributableToOwnersOfParent / NetAssets / TotalAssets, contexts
    CurrentYear{Duration,Instant}, consolidated preferred. Column mapping resolved
    from header names (defensive), fail-closed per element.
  - `FundamentalRecord` (symbol, doc_id, doc_type_code, fiscal_period_end,
    published_ts, values…, source) + idempotent sqlite upsert.
- Storage: `data/raw/htr_fundamentals.db` (HTR-owned; NEVER writes into v5's DB).
- `tools/backfill_edinet_fundamentals.py`: date-range walker; filters
  docTypeCode ∈ {120 annual, 160 semiannual} + secCode present + xbrl/csv flag;
  skips doc_ids already stored (resumable); throttled; JSONL progress log under
  `reports/observability/`. Rollout: 2026 filing season first (May–Jul 2026 =
  FY2026-03 cross-section), then walk back year by year to ~2016.
- Verdict: factor-zoo v2 (value_bp, quality_roe/roa/margin) on the real panel
  through the shared acceptance test. Kill: DSR < 0.95 on the full panel.

## Lane A — Guidance-revision magnitude engine (NEXT, free)

- Corpus rows carry document URLs; revision magnitude = parse 修正前/修正後 tables
  (XBRL attachment when present, else PDF table extraction) → surprise = % revision
  of OP/NP guidance. Event study on ≥2yr (Yanoshin backfill). Kill: IC ≤ 0 or DSR < 0.95.
- Explicitly supersedes the failed title-regex attempt (P19-04 diagnosis).

## Lane C — Forecast/consensus revision momentum (paid, owner decision)

- 四季報 online or J-Quants paid tier. Deferred until A/B verdicts.

## Lane D — Owner's personal calibration (free, behavioral)

- Every trade idea logged pre-trade via the existing predictions/journal path;
  at n ≥ 50, per-setup win rate with CI on the dashboard (descriptive, Rule 11.11.6).

## Order of execution

1. (today) Lane B module + tests + live single-doc probe + background backfill slice #1.
2. Lane A magnitude parser prototype on local corpus.
3. Factor-zoo v2 after backfill slice #2 (≥5 cross-sections).
4. Lane C decision ask only after A/B verdicts exist.
