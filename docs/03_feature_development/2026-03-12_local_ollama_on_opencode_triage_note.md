# 2026-03-12 Local Ollama On OpenCode Triage Note

- Date: 2026-03-12
- Scope: verify whether `opencode` inside `worker-coder` can use the local Ollama service as the stable execution lane
- Status: FIXED
- Conclusion: `LOCAL_OLLAMA_REACHABLE`, `OPENCODE_LOCAL_OLLAMA_CONFIGURED`, `LIVE_PROBE_PASS`

---

## 1. Question

Can the current `worker-coder` runtime use:

- `opencode`
- with model `ollama/glm-4.7-flash:latest`
- against the local Ollama service at `http://host.docker.internal:11434`

as the stable execution lane for coding recovery?

---

## 2. Verified Facts

### Fact A: worker-coder container has the required local-model environment

Observed in `nexus-worker-coder` container:

- `OLLAMA_BASE_URL=http://host.docker.internal:11434`
- `OLLAMA_API_KEY=ollama-local`
- `OPENCODE_BIN=opencode`

### Fact B: local Ollama service is reachable from worker-coder

Direct container probe against:

- `http://host.docker.internal:11434/api/tags`

returned available local models including:

- `glm-4.7-flash:latest`
- `qwq:latest`
- `deepseek-r1:32b`

### Fact C: initial failure was caused by missing opencode provider configuration

Before adding project configuration, running:

- `opencode run 'Reply with exactly OK' --model ollama/glm-4.7-flash:latest`

returned model-not-found and only suggested `ollama-cloud`.

After adding project-level `opencode.json`, running:

- `opencode models`

showed local providers including:

- `ollama/glm-4.7-flash:latest`
- `ollama/deepseek-r1:32b`
- `ollama/qwq:latest`

### Fact D: live probe now passes

Running inside `nexus-worker-coder`:

- `opencode run 'Reply with exactly OK' --model ollama/glm-4.7-flash:latest`

returned:

- `OK`

---

## 3. Operational Conclusion

The issue was not an opencode version-gap finding.

The issue was that the worker runtime had not been given the provider configuration required for local Ollama exposure.

Current operational interpretation:

- local Ollama is reachable and healthy
- `glm-4.7-flash:latest` is present locally
- `opencode + local ollama` is now usable in the current runtime
- project-level `opencode.json` is currently the effective fix

---

## 4. Decision

It is now valid to use local Ollama as the stable execution lane for coding recovery.

Immediate posture:

- keep `stable_local_lane = opencode + ollama/glm-4.7-flash:latest`
- preserve explicit artifacting of `execution_lane`, `model_provider`, and `model_name`
- continue treating `Qwen on opencode` as a separate provider triage path

---

## 5. Next Recommended Step

The next practical step is:

1. run the minimal stable-lane worker-coding cohort
2. verify artifacts record:
   - `execution_lane=stable_local_lane`
   - `model_provider=opencode`
   - `model_name=ollama/glm-4.7-flash:latest`
3. then proceed to authoritative revalidation if the minimal cohort is green

---

## 6. Final Verdict

Recommended triage label:

`FIXED - MISSING OPENCODE LOCAL OLLAMA CONFIGURATION`
