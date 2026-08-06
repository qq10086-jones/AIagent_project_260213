# Evidence-Based Retrospective Remediation Design

Date: 2026-08-06  
Revision: 2 - post-review corrections
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
- the 2026-08-05 exposure snapshot still includes the sold 8035.T; removing it at the review's provisional JPY 54,990 reference leaves approximately 0.415x exposure, far below the declared 1.2x lower band, with no dated resolution path;
- `tools/risk_mandate_snapshot.py::_flag_ages` counts trace rows rather than distinct JPX sessions and appends duplicate same-`asof` rows, so repeated daily runs inflate sunset ages;
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

The current arithmetic also needs a single calendar-aware recomputation pass:

- `JPY 62,800 - JPY 54,990 = JPY 7,810`, not approximately JPY 7,700;
- 2026-07-24 through 2026-08-04 contains eight JPX sessions inclusive and seven elapsed sessions after the trigger day, not nine;
- 2026-07-13 through 2026-08-03 contains fifteen JPX sessions inclusive and fourteen elapsed sessions after creation because 2026-07-20 was closed for Marine Day.

Every queue-age metric must define whether the creation/trigger session is age zero. HTR will use `elapsed eligible sessions after creation`, so a newly created item has age zero; inclusive counts may be shown separately but must be labelled.

Required decomposition:

- decision price: close that first triggered the binding rule;
- compliant execution reference: next eligible broker matching session;
- actual execution price and fee: journal evidence;
- delay cost: actual minus compliant reference, signed by side;
- opportunity cost: quantity not executed, if applicable;
- data status: `provisional` until the fill is journaled.

This follows Perold's implementation-shortfall principle: compare the paper/decision portfolio with the portfolio actually implemented, rather than treating a later close as the executed price.

### 4.2 Correct the broken session-age counter

The current `_flag_ages` implementation is not a trading-session counter. It increments once per trailing JSONL row. The trace contains duplicate rows for 2026-07-28 and 2026-07-29, causing `exit_triggered` to surface as seven sessions on 2026-07-30 when only five distinct JPX sessions had occurred inclusive of the trigger date. The correct inclusive count is nine on 2026-08-05; the trace reports eleven.

Required remediation:

- collapse history to one effective record per `asof` before computing age;
- make same-`asof` snapshot writes idempotent, or define an append-only correction record that supersedes the prior row without adding a session;
- treat a missing eligible-session row as `unobserved`, not `closed`: it neither increments nor resets age; only an explicit observed row without the flag ends continuity;
- surface and persist degraded-history diagnostics whenever a missing or malformed row is encountered in the consumed history;
- validate only the covered JPX path needed by the current computation rather than allowing an unrelated invalid or future row to disable all history;
- if the current `asof` is outside calendar coverage, suppress age escalation for that run and emit a visible warning rather than failing silently;
- recompute all retrospective delay and sunset tables from the corrected definition;
- label the hypothesis that the retrospective copied an inflated trace value as `INFERRED`, because the repository does not preserve provenance proving that copy path.

### 4.3 Separate three scorecards

The revised review must not combine account return, research quality, and system execution into one grade.

| Scorecard | Required metrics | Empty-state rule |
|---|---|---|
| Account outcome | NAV return, benchmark return, active return, drawdown | unavailable if ledger is unreconciled |
| Research validity | live date clusters, trial count, DSR/PBO, cost hurdle, promotion verdict | `insufficient`, never zero-as-failure |
| Execution reliability | open-item age, trigger-to-seen, trigger-to-terminal, band compliance, ledger lag | N/A when denominator is zero |

The separation protects against outcome bias: process quality is judged using information available at decision time, while financial results remain visible and are not excused by process quality.

### 4.4 Correct the 2026-08-26 language

2026-08-26 is the earliest planned 63D review point, not a guaranteed binary verdict date. The allowed verdict set is:

- `confirm`: all locked Rule 16.6 promotion requirements pass;
- `fail`: a predeclared kill criterion fails with adequate evidence;
- `insufficient`: the minimum evidence needed to decide is absent; collection continues without changing capital.

The review must list the frozen trial family, effective date-cluster count, cost model, purge/embargo protocol, DSR, PBO/CPCV, and Harvey-style t-stat hurdle. A date cannot override a failed or immature gate.

### 4.5 Split the Sleeve B experiment

Sleeve B currently has zero deployed capital. Therefore two different questions must be separated:

1. `signal_verdict`: whether E/P survives the live Rule 16.6 research gate;
2. `deployment_verdict`: whether a real B portfolio produced execution, slippage, adherence, and holding-period evidence.

The first may eventually be answered from the live log. The second is `not_started` until a B fill exists. `unwind_to_A` is non-operative when B has no position and must not be reported as a completed pre-commitment response.

### 4.6 Correct the standing-order recommendation

The current P1a recommendation conflicts with P27/Rule 17 precedence.

- Sleeve A is governed by portfolio beta-exposure bands. Do not reintroduce a generic per-symbol stop or bracket.
- Sleeve B is governed by the locked experiment lifecycle unless a separately approved risk exception exists.
- Sleeve C may use a declared per-position bilateral bracket.
- S-kabu cannot use limit, stop, OCO, or multi-day standing orders; it uses eligible market-order matching sessions.
- Whole-lot broker orders must record broker-supported type, expiry date, and renewal state. "Standing" must not imply permanence.

No broker-order identifier, account field, or submission control enters HTR. The system records owner-reported standing-order metadata only; Rule 3 remains intact.

### 4.7 Downgrade the Kelly floor-probability claim

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

The error is explicit in ADR-0012 line 34: it states `lambda * f* approximately 1.4x`, while the recorded inputs give `0.75 * 1.6975 = 1.2731x`. The correction must cite that line directly rather than describing only a conflict among configuration fields.

The review must also state that the approximation omits parameter uncertainty, jumps, discrete rebalancing, execution failure, fees, taxes, and leveraged-ETF path dependence. `market_value x leverage_factor` remains acceptable as an instantaneous exposure proxy, but it is not a long-horizon wealth-process model.

The first decision does not require a simulation project. Under the ADR's own GBM assumptions, expected log growth is approximately 4.376% at 1.2731x and 4.525% at 1.4x: a difference of about 0.148 percentage points, or JPY 575 per year on the 2026-08-05 stale NAV, while the stated floor-hit approximation rises by about 3.95 percentage points. These are model outputs, not forecasts.

LETF cost must use the declared allocation rather than applying 2x drag to all Sleeve A capital. ADR-0012 allocates about JPY 42k to unlevered 1306.T and JPY 175k to 1568.T. At sigma=18%, the analytic variance-drag term is about 3.24% per year on the leveraged component; adding the reviewer's 0.75% fee assumption gives illustrative annual drag of about JPY 7.0k, not JPY 8.7k, or about one-third of the ADR's JPY 21.6k gross equity-premium estimate on JPY 392k beta-adjusted exposure. The memo must verify the official fee as of its date and must not present this approximation as realized tracking difference.

Parameter uncertainty remains decision-relevant: with sigma=18% and even thirty independent annual observations, `SE(mu_hat)=sigma/sqrt(T)` is about 3.29 percentage points and an illustrative normal 95% interval for full Kelly is approximately [-0.29, 3.69]. This does not identify the true premium, and serial dependence would further weaken the independent-sample calculation. Because the floor formula is written in terms of fractional Kelly, uncertainty in `mu/sigma^2` re-enters when a fixed 1.4x target is mapped to lambda.

### 4.8 Close the out-of-band third state

After provisional removal of 8035.T from the 2026-08-05 stale snapshot, beta-adjusted exposure is JPY 159,586. Replacing its stale JPY 58,550 mark with the retrospective's JPY 54,990 reference gives provisional NAV JPY 384,321, exposure approximately 0.415x, and a gap of approximately JPY 301,599 to the declared 1.2x lower band.

`0.055 * JPY 301,599 = approximately JPY 16,588 per year` is a **model-implied gross opportunity cost under the mandate's equity-premium assumption**, not an observed loss. It excludes LETF drag, fees, timing, parameter uncertainty, and the possibility that the mandate itself should be changed. Its value is diagnostic: the recurring unresolved mandate state is potentially more material than the one-off 8035.T shortfall and therefore requires an explicit owner decision.

Within one dated decision record, the owner must choose one of:

1. deploy toward the existing band under the existing execution rules;
2. propose a Rule 4 amendment to the target/band and record old value, new value, rationale, expected impact, and verification;
3. approve a time-bounded exception with rationale, review date, and hard expiry.

Silence or an indefinitely open exception is not a fourth state. The system remains advice-only and cannot choose or execute any option for the owner.

### 4.9 Narrow causal claims and propagate retractions

The following language changes are required:

- "the bracket caused the exit" -> "the written bracket and repeated surfacing are consistent with contributing to the exit";
- "the delay caused JPY X loss" -> "estimated shortfall versus the predeclared compliant reference" until actual fill reconciliation;
- "negative knowledge saved money" -> "the rejected signals would have had negative average forward rank association in the observed window; avoided P&L is not identifiable without a frozen execution policy";
- "the system protected NAV" -> separate declared policy compliance from a favorable outcome caused by under-deployment.

The withdrawn statement that low exposure provided a "second empirical protection" still appears in `PROJECT_STATUS.md`. The revised retrospective must not merely retract it locally: the current-state summary and the original live status entry must be annotated as superseded, preserving audit history while preventing the withdrawn interpretation from continuing as current truth.

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

### P28 - Ledger, Retrospective, and Mandate-State Closure

Purpose: establish a trustworthy as-of snapshot before judging performance.

Acceptance:

- 8035.T actual sell fill is recorded through the existing Section 14 record-only path;
- no stale 8035.T holding remains in the next risk snapshot;
- NAV, realized/unrealized P&L, benchmark return, and active return reconcile to a documented tolerance;
- implementation shortfall reports decision, eligible execution, actual execution, fees, and provisional/final status;
- all delay tables are recomputed with the age-zero JPX-session convention and distinguish inclusive from elapsed counts;
- after reconciliation, exposure and band status are recomputed without 8035.T;
- a dated owner decision closes the band breach by deployment, a Rule 4 band proposal, or a time-bounded exception with expiry; P28 does not select or execute the option;
- the withdrawn low-exposure interpretation is marked superseded where it remains live in `PROJECT_STATUS.md`;
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
- `_flag_ages` uses distinct covered JPX `asof` sessions, same-date runs do not increase age, and same-date snapshot persistence is idempotent or explicitly superseding;
- missing observations do not increment or reset age, explicit flag absence resets it, and all degraded/calendar-uncovered states are visible in stdout and trace diagnostics;
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

### P32 - Risk-Mandate Decision Memo

Purpose: give the owner a bounded, auditable decision without turning an arithmetic inconsistency into a research programme.

Acceptance:

- produce one short derivation memo reproducing the ADR arithmetic, including the explicit line-34 error;
- show the 1.2731x-versus-1.4x log-growth difference and floor-hit trade-off as model outputs;
- apply LETF variance drag and the as-of verified official fee only to the planned leveraged component, and distinguish analytic drag from observed tracking difference;
- show a transparent parameter-uncertainty sensitivity, including the assumptions behind any interval;
- present at least three owner alternatives: retain 1.4x and withdraw the 10% claim; align the target with the stated fractional-Kelly bound; or abandon the Kelly provenance and re-justify the band as an owner preference;
- allow a time-bounded deferral only through the P28 exception path;
- output proposal alternatives only; do not edit the active mandate.

Block bootstrap and jump/regime Monte Carlo are explicitly deferred. They become a separate proposed research task only if the owner chooses a quantitatively risk-calibrated band, intends to occupy it, and first states what decision the simulation could change.

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
P32 decision memo ------------> possible future Rule 4 proposal

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
- direct regression tests for `_flag_ages`: duplicate same-`asof`, non-trading dates, missing dates, flag close/reopen, and corrupted history;
- an integration test proving repeated snapshot generation for one `asof` does not advance sunset age;
- fixture tests for the age-zero trading-session convention and S-kabu eligibility;
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

## 11. Review-Disposition Record

| Review item | Disposition | Reason |
|---|---|---|
| Missing out-of-band state closure | Accept | The existing backlog reconciled the ledger but did not force a governed mandate-state decision. |
| `_flag_ages` counts rows | Accept | Direct code and trace evidence confirm duplicate-date inflation and early sunset escalation. |
| Retrospective copied the bad counter | Accept as inference only | The number is inconsistent with the JPX calendar, but the repository does not prove its provenance. |
| Reduce P32 scope | Accept | The immediate inconsistency is arithmetic and owner-governance work; unconditional simulation has poor decision value. |
| JPY 8.7k drag / 36% of edge | Modify | It incorrectly applies leveraged-fund costs to all JPY 217k; the declared mix has only about JPY 175k in 1568.T. |
| Owner binary choice | Modify | At least three coherent governance alternatives exist, plus a time-bounded exception; none may be silently activated. |
| Arithmetic/session/status corrections | Accept | Repository arithmetic, the covered JPX calendar, and the live status text support the corrections. |
| Missing trace row resets age | Reject reset semantics | Missing means unobserved, not closed. Skip without incrementing, retain prior observations, and mark the age degraded; only explicit absence resets continuity. |
| Any invalid row disables all age history | Reject | Validation must be scoped to the consumed path and every degradation must be visible; unrelated future rows cannot poison current escalation. |
