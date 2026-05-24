# P9-04 Human Alerts Design

## Goal

Generate local, human-readable alert records when a watched candidate crosses a staged buy, stop, or sell level. Alerts are research-only notifications and must never place, simulate, or prepare orders.

## Scope

P9-04 consumes the existing candidate/ladders shape used by the dashboard. It compares a current market price against the seven Rule 9.3 ladder levels and emits alert records for crossed levels:

- entry levels trigger when `current_price <= level_price`
- stop triggers when `current_price <= stop_price`
- exit levels trigger when `current_price >= level_price`

## Design

Add `hot_theme_rotator.alerts.human_alerts` with:

- `AlertRecord`: immutable alert payload with `alert_id`, `symbol`, `level_id`, `level_price`, `current_price`, `direction`, `severity`, `reason`, `risk_warning`, `data_ts`, and `research_only`.
- `AlertThrottle`: in-memory duplicate guard keyed by `(symbol, level_id, trade_date)`.
- `build_ladder_alerts`: pure function that returns alert records for a symbol/ladder/current price.

`alert_id` is deterministic from `symbol`, `level_id`, `trade_date`, and `data_ts`. The first implementation does not send email, desktop notifications, chat messages, or UI popups. It only produces records that later surfaces can render.

## Advice-Only Boundary

Alerts must not include order side, quantity, notional, broker, account, route, or submit payload fields. The record carries `research_only=True` and a plain `risk_warning`.

## Non-Goals

- No external notification channel.
- No paper trade creation.
- No broker integration.
- No UI changes in this cycle.
