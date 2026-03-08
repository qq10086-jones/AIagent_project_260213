# Orchestrator Internal Layer Map
## Date: 2026-03-07
## Status: Draft — to be finalized in WS-11-01

---

## 1. Target 4-Layer Structure

```
Layer 1: Transport/Adapter Layer   (src/adapters/)
Layer 2: Service Layer             (src/vnext/)
Layer 3: Domain Layer              (src/)
Layer 4: Infrastructure Layer      (src/data/, src/infra/)
```

---

## 2. Current File → Target Layer Mapping

### Layer 1 — Transport/Adapter (Target: src/adapters/)

| Current Location | Target | Notes |
|-----------------|--------|-------|
| index.js (Discord Client) | src/adapters/discord_gateway.js | Extract Discord.js, event handlers, replyChunked, safeTranslate |
| index.js (HTTP route definitions) | index.js (thin, ≤800 lines) | Keep only route wiring; move all inline logic out |
| index.js (cron jobs) | src/adapters/cron_scheduler.js | Move node-cron setup out of index.js |

---

### Layer 2 — Service Layer (Current: src/vnext/)

| File | Status | Notes |
|------|--------|-------|
| src/vnext/brain_router.js | OK | Add policy override call (WS-13) |
| src/vnext/brain_router_policy.js | MISSING | Create in WS-13 |
| src/vnext/input_normalizer.js | OK | — |
| src/vnext/task_envelope.js | OK | — |
| src/vnext/runtime_dispatch.js | OK | — |
| src/vnext/chat_entrypoint.js | OK | — |
| src/vnext/approval_entrypoint.js | OK | — |
| src/vnext/approval_interceptor.js | OK | — |
| src/vnext/workflow_notification_delivery.js | OK | — |
| src/vnext/workflow_runtime_notifier.js | OK | — |
| src/vnext/risk_classifier.js | OK | — |
| src/vnext/observability_reporter.js | OK | — |
| src/vnext/discord_reply_adapter.js | OK | — |
| src/vnext/route_contract.js | OK | — |
| src/vnext/dispatch_contract.js | OK | — |
| src/vnext/response_protocol.js | OK | — |
| src/vnext/contract_validator.js | OK | — |
| src/vnext/coder_directive.js | OK | — |
| src/vnext/artifact_timeline.js | OK | — |
| src/vnext/tool_permission_guard.js | OK | — |

---

### Layer 3 — Domain Layer (Current: src/)

| File | Status | Notes |
|------|--------|-------|
| src/workflow_engine.js | VIOLATION (2131 lines) | Decompose into workflow_runner, workflow_state, workflow_artifact_audit |
| src/domain/workflow_runner.js | MISSING | Create in WS-11-04 |
| src/domain/workflow_state.js | MISSING | Create in WS-11-04 |
| src/domain/workflow_artifact_audit.js | MISSING | Create in WS-11-04 |
| src/domain/memory_reader.js | MISSING | Create in WS-15-02 |
| src/domain/memory_writer.js | MISSING | Create in WS-15-04 |
| src/coding_team_validators.js | OK | — |
| src/coding_team_handoff_validators.js | OK | — |
| src/artifact_registry.js | OK | — |
| src/artifact_pack_validator.js | OK | — |
| src/final_result_packager.js | OK | — |
| src/prompt_script_registry.js | OK | — |
| src/agent_contract_registry.js | OK | — |
| src/handoff_contract_registry.js | OK | — |
| src/tool_adapter_registry.js | OK | — |
| src/registry.js | OK | — |
| src/qa_verifier.js | OK | — |
| src/schema_lite_validator.js | OK | — |
| src/patch_manager.js | OK | — |
| src/policy.js | OK | — |
| src/coding_execution_adapters.js | OK | — |
| src/coding_executor.js | OK | — |
| src/ingress.js | REVIEW | Check if still needed after Layer 1 extraction |
| src/nlp/router.js | REVIEW | LLM classification only; policy override moves to Layer 2 |

---

### Layer 4 — Infrastructure Layer (Target: src/data/, src/infra/)

| Current Location | Target | Notes |
|-----------------|--------|-------|
| index.js (pool.query calls) | src/data/task_repository.js | All task SQL |
| index.js (pool.query calls) | src/data/run_repository.js | All run/workflow SQL |
| index.js (pool.query calls) | src/data/event_repository.js | All event SQL |
| index.js (pool, redis, s3 init) | src/infra/connections.js | Shared connection objects |
| index.js (Redis operations) | src/data/stream_repository.js | Redis stream operations |

---

## 3. Cross-Layer Violations (Current)

| Violation | Severity | Fix |
|-----------|----------|-----|
| index.js imports Discord.js (Layer 1 duty in Layer 1 file, but mixed with Layer 3 logic) | High | WS-11-02 |
| index.js has raw pool.query (Layer 4 in Layer 1 file) | High | WS-11-03 |
| index.js calls qwenChat directly (Layer 3/4 in Layer 1 file) | High | WS-11-05 |
| index.js wires Discord event + business logic inline | High | WS-11-02 |
| workflow_engine.js is 2131 lines mixing runner + state + artifact logic | Medium | WS-11-04 |

---

## 4. Layer Boundary Rules (Enforcement)

```
Layer 1 → MAY import from: Layer 2
Layer 1 → MUST NOT import from: Layer 3, Layer 4

Layer 2 → MAY import from: Layer 3
Layer 2 → MUST NOT import from: Layer 1, Layer 4 (raw DB/Redis)

Layer 3 → MAY import from: Layer 4 (via repository interfaces)
Layer 3 → MUST NOT import from: Layer 1, Layer 2

Layer 4 → MUST NOT import from: Layer 1, Layer 2, Layer 3
         (Layer 4 exports only — no upward imports)
```

---

## 5. Complexity Budget Status

| Module | Current Lines | Budget | Status |
|--------|--------------|--------|--------|
| src/index.js | 2790 | 800 | OVER by 1990 |
| src/workflow_engine.js | 2131 | 600 | OVER by 1531 |
| src/vnext/brain_router.js | 173 | 300 | OK |
| src/vnext/runtime_dispatch.js | 156 | 300 | OK |
| src/vnext/chat_entrypoint.js | 235 | 300 | OK |
| src/vnext/approval_entrypoint.js | 147 | 300 | OK |

---

## 6. Post-M3 Target State

After Milestone 3 decomposition:

```
src/
├── adapters/
│   ├── discord_gateway.js      (Discord events, <400 lines)
│   └── cron_scheduler.js       (Cron jobs, <100 lines)
├── data/
│   ├── task_repository.js      (<250 lines)
│   ├── run_repository.js       (<250 lines)
│   ├── event_repository.js     (<250 lines)
│   └── stream_repository.js    (<200 lines)
├── domain/
│   ├── workflow_runner.js      (<300 lines)
│   ├── workflow_state.js       (<200 lines)
│   ├── workflow_artifact_audit.js (<150 lines)
│   ├── memory_reader.js        (<100 lines)
│   └── memory_writer.js        (<100 lines)
├── infra/
│   └── connections.js          (<50 lines, pool/redis/s3 init)
├── vnext/                      (service layer, no change to structure)
│   └── brain_router_policy.js  (NEW, <150 lines)
├── workflow_engine.js           (<600 lines, delegates to domain/)
└── index.js                     (<800 lines, HTTP routes only)
```
