# P8-05 Ladder Feedback Design

## Goal

Build an opportunity-ladder feedback evaluator on top of the existing P9-01 decision log, P9-02 outcome join, and P9-03 calibration engine.

## Scope

P8-05 consumes existing `PredictionRecord` and `OutcomeRecord` objects. It does not create a new storage format, does not recompute outcomes, and does not publish any order or alert. The evaluator reports evidence for the seven Rule 9.3 ladder tiers:

- `aggressive_entry`
- `balanced_entry`
- `conservative_entry`
- `stop_price`
- `first_exit`
- `second_exit`
- `stretch_exit`

## Design

Add `hot_theme_rotator.calibration.ladder_feedback` with two immutable report dataclasses:

- `LadderTierFeedback`: tier name, direction, sample count, touched count, optional touch rate, and calibration status.
- `LadderFeedbackReport`: trade date range, total matched complete samples, per-tier feedback, and the existing bullish calibration report for 3D returns.

The evaluator pairs predictions and outcomes by `prediction_id`, keeps only opportunity predictions that carry a full ladder in `prediction.extra["ladder"]`, and keeps only outcomes with `status == "complete"`. It fails closed when a complete matched outcome is missing a required tier or a tier payload does not contain a boolean `touched` value.

## Calibration Boundary

Tier touch rate is level-touch evidence, not a win rate. If a tier has fewer than `min_samples` complete matched outcomes, its `touch_rate` remains `None` and its status is `insufficient_calibration`. Only when `sample_count >= min_samples` may the evaluator expose numeric `touch_rate`.

The existing P9-03 `build_calibration_report` remains the only component that emits Brier score, log loss, and reliability bins for bullish 1D/3D/5D outcome calibration.

## Non-Goals

- No UI changes in this cycle.
- No alerts.
- No paper trading.
- No broker or execution integration.
- No new score status values.
