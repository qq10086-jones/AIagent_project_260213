# ADR-0002: Rotation Signal

## Status

Accepted

## Context

HotThemeRotator currently emits advice-only `BUY`, `HOLD`, `TAKE_PROFIT`,
`STOP_LOSS`, and `NO_TRADE` signals. `ROTATE` is reserved in the schema but
was intentionally left out of the P4-01 MVP because it needs a clear contract:
it compares an existing holding with a stronger new theme leader, so a vague
implementation could accidentally encourage churn.

The project remains advice-only. `ROTATE` is a human-readable recommendation,
not an order instruction.

## Decision

`ROTATE` applies only when all of these conditions hold:

- There is an existing position.
- The existing position has not reached take-profit.
- The existing position has not reached stop-loss.
- The market and external temperature do not block trading.
- The candidate leader passes the normal entry thresholds.
- The candidate is from a different theme than the current holding.
- The candidate leader score exceeds the current holding leader score by at
  least `rotate_min_score_delta`.

Priority order is:

1. `TAKE_PROFIT`
2. `STOP_LOSS`
3. `ROTATE`
4. `HOLD`
5. new-entry `BUY` / `NO_TRADE`

This means explicit exit risk and profit-taking always outrank rotation. A weak
candidate or same-theme candidate does not become `ROTATE`; the existing
position remains `HOLD`.

## Consequences

- The signal engine needs current-holding comparison fields in `SignalInput`.
- `ROTATE` remains blocked by market and external risk blocks.
- Risk governor sizing remains outside the signal engine. The daily pipeline can
  still convert blocked buy-like advice to `NO_TRADE` before rendering.
- Backtests must include costs before treating rotation results as anything more
  than a research draft.
