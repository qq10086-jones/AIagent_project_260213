# ADR-0001: Independent HotThemeRotator Project Spine

Date: 2026-05-18

## Status

Accepted

## Context

Existing work is split between `Project_optimized` and `Project_v5`. Those projects contain useful components, but also carry historical strategy decisions, production reports, paper state, old experiments and encoding issues. Continuing to modify them directly would make it hard to separate the user's current hotspot rotation strategy from previous ETF and sprint strategy work.

## Decision

Create `quant_trading/HotThemeRotator` as an independent project folder. Use it as the new source of truth for the market-temperature and hot-theme leader rotation tool.

`Project_optimized` and `Project_v5` become reference sources. Migration must happen through explicit tasks and adapters.

## Consequences

- New work has a clean architecture and governance spine.
- Old project state remains untouched.
- Initial delivery is slower than editing old scripts directly, but future changes become easier to reason about.
- The project starts advice-only and cannot auto-trade without a future governance decision.

