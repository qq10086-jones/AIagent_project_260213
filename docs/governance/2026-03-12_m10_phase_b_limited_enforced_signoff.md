# M10 Phase B Limited Enforced Sign-off

- Date: 2026-03-12
- Scope: M7/M10 controlled enablement transition from advisory-only to limited enforced execution
- Reviewers: PM + Architect
- Status: APPROVED

---

## Decision

Approve entry into **Phase B limited enforced execution** for the narrow cohort:

- `workflow_type = coding_team_v0`
- `project_type = webapp_crm`
- `input_class = fe_led`

Approved runtime posture:

- `master_enabled = true`
- `dynamic_routing_enabled = true`
- `router_mode = dynamic_routing_enforced`
- cohort restriction remains enabled

This approval is explicitly:

- limited to the approved cohort only
- reversible through existing rollback controls
- conditioned on preserving static-policy fallback for unavailable / low-confidence / denied classifier cases

---

## Evidence Basis

The following evidence was reviewed before approval:

1. Advisory evidence base remains above the original entry threshold:
   - observed routing samples: `89`
   - classifier-backed `gated_parallel_allowed`: `71`
   - forced sequential: `18`
2. Enforced Canary A passed:
   - low-risk pure UI cases succeed in enforced mode
3. Enforced Canary B passed:
   - FE + BE queue in parallel
   - `target_paths` are strictly disjoint
4. Observability correlation canary passed:
   - `routing_decision_log`
   - `waterfall_stage_log`
   - derived `branch_completion_be` / `branch_completion_fe`
   can all be correlated by the same parent `run_id`
5. Promotion safety prerequisites are present:
   - promotion conflict detection landed
   - atomic apply + rollback journaling landed

---

## Rollback Authority

Rollback remains operator-simple and unchanged:

1. set `router_mode=static_policy_only`
2. if needed, set `dynamic_routing_enabled=false`
3. if needed, set `force_sequential=true`

No cohort widening, no high-risk work-shape enablement, and no fallback disabling is authorized by this sign-off.

---

## Next Approved Step

The next approved execution step is:

- `T-23c`: full `fe_safe` DAG canary under limited enforced mode

This is allowed because:

- Phase B limited enforced entry has now been explicitly recorded
- the current cohort remains narrow
- observability and rollback controls remain intact
