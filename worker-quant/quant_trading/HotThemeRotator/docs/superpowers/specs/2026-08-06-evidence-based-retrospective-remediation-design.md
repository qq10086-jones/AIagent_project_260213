# Evidence-Based Retrospective Remediation Design

Date: 2026-08-06  
Status: PROPOSED - approved for documentation design, not approved for parameter activation  
Scope: `docs/proposals/retrospective_review_2026-08-04.md`, proposed P28-P33 additions to `docs/01_TASKS.md`, and a current-state refresh in `PROJECT_STATUS.md`  
Non-scope: broker automation, live-order routing, risk-mandate parameter changes, signal promotion, and capital deployment

## 1. Decision

Use an evidence-matrix structure for the retrospective and improvement backlog. Every material conclusion must identify:

1. the project-local observation and its as-of date;
2. the external theory or empirical literature that motivates the interpretation;
3. the inference boundary - what the evidence does not establish;
4. a falsifiable acceptance criterion;
5. the proposed task that closes the loop.

External literature supplies mechanisms and validation methods. It does not prove that a mechanism is effective in HTR. HTR's forward logs, append-only journal, execution records, and rule traces remain the deciding evidence.

All proposed changes remain proposals under Rule 13.1. Any parameter activation or change to `configs/risk_mandate.json` requires owner approval and the Rule 4 record: field, old value, new value, reason, expected impact, and verification.

## 2. Why the Current Review Needs Remediation

The 2026-08-04 retrospective is unusually strong on negative evidence, outcome-bias correction, and execution diagnosis. It is not yet a closed project artifact because:

- it is untracked and marked `REVIEW`;
- the project task ledger ends at P27;
- `PROJECT_STATUS.md` still exposes stale July current-state and next-action snapshots;
- the reported 8035.T exit has not entered the Section 14 journal, so final NAV and execution shortfall are unresolved;
- several recommendations conflict with binding rules or with their own statistical assumptions;
- some causal language is stronger than the available counterfactual evidence supports.

The remediation must preserve the review's central conclusion: engineering and falsification quality are strong, while demonstrated trading edge and execution closure are weak.

## 3. Evidence Classification

Each material statement in the revised review receives one of four labels.

| Label | Meaning | Permitted language |
|---|---|---|
| `OBSERVED` | Directly read from journal, snapshot, or append-only trace | "recorded", "measured as of" |
| `DERIVED` | Reproducible arithmetic from observed values | "computed under" |
| `INFERRED` | Mechanistic interpretation supported by theory or comparison | "consistent with", "suggests" |
| `PROPOSED` | Future change awaiting Rule 4/13 approval | "proposed", "would be accepted if" |

Counterfactual amounts, including avoided losses and implementation shortfall before the actual fill is recorded, cannot be labelled `OBSERVED`.

## 4. Required Corrections to the Retrospective

### 4.1 Accounting and execution shortfall

The review's `approximately JPY 7,700` delay cost is a scenario estimate, not a final realized amount. The compliant reference point, the actual execution price, fees, and S-kabu matching session must be shown separately.

Required decomposition:

- decision price: close that first triggered the binding rule;
- compliant execution reference: next eligible broker matching session;
- actual execution price and fee: journal evidence;
- delay cost: actual minus compliant reference, signed by side;
- opportunity cost: quantity not executed, if applicable;
- data status: `provisional` until the fill is journaled.

This follows Perold's implementation-shortfall principle: compare the paper/decision portfolio with the portfolio actually implemented, rather than treating a later close as the executed price.

### 4.2 Separate three scorecards

The revised review must not combine account return, research quality, and system execution into one grade.

| Scorecard | Required metrics | Empty-state rule |
|---|---|---|
| Account outcome | NAV return, benchmark return, active return, drawdown | unavailable if ledger is unreconciled |
| Research validity | live date clusters, trial count, DSR/PBO, cost hurdle, promotion verdict | `insufficient`, never zero-as-failure |
| Execution reliability | open-item age, trigger-to-seen, trigger-to-terminal, band compliance, ledger lag | N/A when denominator is zero |

The separation protects against outcome bias: process quality is judged using information available at decision time, while financial results remain visible and are not excused by process quality.

### 4.3 Correct the 2026-08-26 language

2026-08-26 is the earliest planned 63D review point, not a guaranteed binary verdict date. The allowed verdict set is:

- `confirm`: all locked Rule 16.6 promotion requirements pass;
- `fail`: a predeclared kill criterion fails with adequate evidence;
- `insufficient`: the minimum evidence needed to decide is absent; collection continues without changing capital.

The review must list the frozen trial family, effective date-cluster count, cost model, purge/embargo protocol, DSR, PBO/CPCV, and Harvey-style t-stat hurdle. A date cannot override a failed or immature gate.

### 4.4 Split the Sleeve B experiment

Sleeve B currently has zero deployed capital. Therefore two different questions must be separated:

1. `signal_verdict`: whether E/P survives the live Rule 16.6 research gate;
2. `deployment_verdict`: whether a real B portfolio produced execution, slippage, adherence, and holding-period evidence.

The first may eventually be answered from the live log. The second is `not_started` until a B fill exists. `unwind_to_A` is non-operative when B has no position and must not be reported as a completed pre-commitment response.

### 4.5 Correct the standing-order recommendation

The current P1a recommendation conflicts with P27/Rule 17 precedence.

- Sleeve A is governed by portfolio beta-exposure bands. Do not reintroduce a generic per-symbol stop or bracket.
- Sleeve B is governed by the locked experiment lifecycle unless a separately approved risk exception exists.
- Sleeve C may use a declared per-position bilateral bracket.
- S-kabu cannot use limit, stop, OCO, or multi-day standing orders; it uses eligible market-order matching sessions.
- Whole-lot broker orders must record broker-supported type, expiry date, and renewal state. "Standing" must not imply permanence.

No broker-order identifier, account field, or submission control enters HTR. The system records owner-reported standing-order metadata only; Rule 3 remains intact.

### 4.6 Downgrade the Kelly floor-probability claim

The ADR's own inputs imply:

```text
mu = 0.055
sigma = 0.18
full Kelly f* = mu / sigma^2 = 1.6975
lambda = 0.75 -> exposure = 1.2731x
declared target = 1.4x -> implied lambda = 0.8247
P(hit 0.25) at lambda 0.75 = 9.92%
P(hit 0.25) at target 1.4x = 13.87%
```

Consequently, `target=1.4x` and `P(floor hit)<=10%` cannot both be presented as outputs of the stated approximation. This is a derivation inconsistency, not authorization to silently change the owner-declared target.

The review must also state that the approximation omits parameter uncertainty, jumps, discrete rebalancing, execution failure, fees, taxes, and leveraged-ETF path dependence. `market_value x leverage_factor` remains acceptable as an instantaneous exposure proxy, but it is not a long-horizon wealth-process model.

### 4.7 Narrow causal claims

The following language changes are required:

- "the bracket caused the exit" -> "the written bracket and repeated surfacing are consistent with contributing to the exit";
- "the delay caused JPY X loss" -> "estimated shortfall versus the predeclared compliant reference" until actual fill reconciliation;
- "negative knowledge saved money" -> "the rejected signals would have had negative average forward rank association in the observed window; avoided P&L is not identifiable without a frozen execution policy";
- "the system protected NAV" -> separate declared policy compliance from a favorable outcome caused by under-deployment.

## 5. Literature-to-Design Matrix

| Evidence | What it supports in HTR | What it does not prove |
|---|---|---|
| Bailey & Lopez de Prado, DSR | Count trials; deflate selected performance | A DSR pass guarantees live profit |
| Bailey et al., PBO | Estimate selection overfit across configurations | Small forward samples become sufficient |
| Harvey, Liu & Zhu | Use a higher multiple-testing hurdle in factor research | t>3 alone authorizes capital |
| Perold, implementation shortfall | Measure decision-to-portfolio execution loss | A later close equals actual execution |
| Odean, disposition effect | Prewritten exits address a documented retail bias | Every delayed loss was disposition effect |
| Gollwitzer & Brandstatter | Specific if-then plans improve action initiation | A JSON rule guarantees execution |
| Baron & Hershey; Aiyer et al. | Evaluate process separately from realized outcome | Outcomes should be ignored |
| Ancker et al. | Repeated alerts can reduce acceptance | Medical effect sizes transfer to trading |
| Busseti, Ryu & Boyd | Constrain drawdown directly and compare via simulation | HTR's GBM assumptions are accurate |
| Avellaneda & Zhang | LETF returns depend on path, variance, and financing | US LETF estimates equal 1568.T parameters |

## 6. Proposed Task Backlog

All tasks below enter `docs/01_TASKS.md` as `proposed - awaiting owner activation`. Documentation fixes may be performed without activating runtime behavior. Any code/config phase requires a later implementation plan and the applicable Rule 4/13 approval.

### P28 - Ledger and Retrospective Baseline Closure

Purpose: establish a trustworthy as-of snapshot before judging performance.

Acceptance:

- 8035.T actual sell fill is recorded through the existing Section 14 record-only path;
- no stale 8035.T holding remains in the next risk snapshot;
- NAV, realized/unrealized P&L, benchmark return, and active return reconcile to a documented tolerance;
- implementation shortfall reports decision, eligible execution, actual execution, fees, and provisional/final status;
- the revised retrospective cites artifact paths and as-of timestamps.

### P29 - Decision Queue and Execution Observability

Purpose: make advice-to-terminal-state delay observable without creating broker execution.

State model:

```text
open -> acknowledged -> executed | declined | expired | superseded
```

Acceptance:

- deterministic advice ID and append-only transition records;
- source rule, created timestamp, trading-session age, severity, and evidence pointer;
- decline requires a structured reason plus optional note;
- afterclose reports open age distribution and terminal-state counts;
- trigger-to-seen and trigger-to-terminal are separately measurable;
- CLI/afterclose recording precedes any UI write path;
- any new HTTP mutation first amends the Rule 11.5 whitelist and receives governance review.

### P30 - Low-Noise State-Transition Notifications

Purpose: surface binding state changes while preventing alert fatigue.

Acceptance:

- one owner-selected channel enabled through Rule 12.7 double confirmation;
- only state transitions notify; unchanged open states do not repeat;
- dedupe key, severity, cooldown, monthly budget, and delivery audit are present;
- notification content links to the decision ID and contains no order control;
- monthly metrics include sent, delivered, acknowledged, duplicate-suppressed, and trigger-to-seen delay;
- automatic rollback to silent mode if error rate or duplicate rate breaches a predeclared threshold.

### P31 - Locked 63D Evidence Review Protocol

Purpose: prevent the calendar from moving the research gate.

Acceptance:

- 2026-08-26 labelled earliest review, not guaranteed verdict;
- three-valued result: confirm/fail/insufficient;
- trial family and all attempted variants counted before computation;
- independent date clusters, raw rows, maturity coverage, and missingness reported;
- PIT, survivorship, cost, purge, embargo, DSR, PBO/CPCV, and t-stat checks emitted;
- E/P and B/P reported independently; B/P sign reversal cannot be hidden by a composite;
- signal and deployment verdicts are separate;
- no capital/config change occurs from the report alone.

### P32 - Risk-Mandate Derivation Audit

Purpose: reconcile owner-declared parameters with the mathematical claim attached to them.

Acceptance:

- independently reproduce all ADR arithmetic;
- identify which combinations of target exposure and fractional Kelly satisfy the stated approximation;
- model 1568.T as a daily-reset LETF, including realized-variance/path term and observed fee structure;
- compare analytic GBM approximation, historical block bootstrap, and Monte Carlo with jumps/regime stress;
- include parameter-uncertainty sensitivity for mu and sigma;
- report floor-hit estimates with assumptions and intervals, never as calibrated probabilities;
- output proposal alternatives only; do not edit the active mandate.

### P33 - Three-Ledger KPI and Rule-Sunset Review

Purpose: make system value and maintenance burden measurable.

Acceptance:

- publish separate account, research, and execution scorecards;
- define every numerator, denominator, unavailable state, and as-of date;
- measure ledger lag and band-compliance only on days with valid prices and reconciled positions;
- scan runtime references from rules to config/code/tests/reports;
- rules unused for six months enter an owner review list, never automatic deletion;
- preserve audit history when rules are merged or retired.

## 7. Dependency Order

```text
P28 ledger closure
  -> P29 decision identity and states
      -> P30 notifications and execution KPIs

P31 evidence protocol --------> possible future signal proposal
P32 mandate audit ------------> possible future Rule 4 proposal

P28 + P29 + P31 + P32
  -> P33 consolidated scorecards and sunset review
```

P28 is first because every performance conclusion depends on a reconciled account. P29 precedes P30 because a notification without a durable decision identity cannot be acknowledged or audited. P31 and P32 are independent research/governance lanes and may be designed in parallel later, but neither may activate capital changes.

## 8. Verification Strategy

Documentation verification:

- scan for unsupported definitive terms: `caused`, `proved`, `saved`, `guaranteed`, and bare `win rate`;
- verify every performance number has an as-of date and source path;
- verify all external studies have DOI or an explicit DOI-pending label;
- verify every cross-domain citation states its transfer limitation;
- verify proposed tasks do not use `done`, `active`, or `accepted` language.

Future implementation verification:

- TDD for queue state transitions, idempotency, and append-only behavior;
- fixture tests for trading-session age and S-kabu eligibility;
- property tests for no duplicate transition notification;
- replay the 8035.T timeline as a golden scenario;
- confirm no broker fields/routes and no change to the Rule 11.5 write surface without explicit governance;
- full non-slow test lane plus targeted integration tests.

## 9. Evidence Register

1. Bailey, D. H., & Lopez de Prado, M. (2014). *The Deflated Sharpe Ratio*. DOI: `10.3905/jpm.2014.40.5.094`.
2. Bailey, D. H., Borwein, J. M., Lopez de Prado, M., & Zhu, Q. J. (2016). *The Probability of Backtest Overfitting*. DOI: `10.21314/JCF.2016.322`.
3. Harvey, C. R., Liu, Y., & Zhu, H. (2016). *... and the Cross-Section of Expected Returns*. DOI: `10.1093/rfs/hhv059`.
4. Perold, A. F. (1988). *The Implementation Shortfall*. DOI: `10.3905/jpm.1988.409150`.
5. Odean, T. (1998). *Are Investors Reluctant to Realize Their Losses?* DOI: `10.1111/0022-1082.00072`.
6. Gollwitzer, P. M., & Brandstatter, V. (1997). *Implementation Intentions and Effective Goal Pursuit*. DOI: `10.1037/0022-3514.73.1.186`.
7. Gollwitzer, P. M. (1999). *Implementation Intentions: Strong Effects of Simple Plans*. DOI: `10.1037/0003-066X.54.7.493`.
8. Baron, J., & Hershey, J. C. (1988). *Outcome Bias in Decision Evaluation*. DOI: `10.1037/0022-3514.54.4.569`.
9. Aiyer, S., et al. (2023). *Outcomes Affect Evaluations of Decision Quality*. DOI: `10.5334/irsp.751`.
10. Ancker, J. S., et al. (2017). *Effects of Workload, Work Complexity, and Repeated Alerts on Alert Fatigue*. DOI: `10.1186/s12911-017-0430-8`.
11. Busseti, E., Ryu, E. K., & Boyd, S. (2016). *Risk-Constrained Kelly Gambling*. DOI: `10.3905/joi.2016.25.3.118`; open preprint: `arXiv:1603.06183`.
12. Avellaneda, M., & Zhang, S. (2010). *Path-Dependence of Leveraged ETF Returns*. DOI: `10.1137/090760805`.

## 10. Explicit Non-Claims

This design does not claim:

- that HTR has demonstrated profitable alpha;
- that E/P will pass on or after 2026-08-26;
- that a notification will cause timely execution;
- that a bracket is optimal for every position;
- that the Kelly approximation is an accurate model of the owner account;
- that 1.27x is the correct replacement for 1.4x;
- that literature effect sizes transfer directly into a Japanese retail trading workflow;
- that engineering quality compensates for negative active return.

The design's success condition is narrower: the revised review becomes internally consistent, empirically traceable, statistically gated, and convertible into a prioritized backlog without silently changing live behavior.
