# 2026-03-10 OpenCode Qwen Runtime Note

## Intent

Stabilize the default coding execution path as:

- provider: `opencode`
- model: `qwen3-coder-plus-2025-07-22`

This keeps Nexus responsible for workflow/orchestration while `opencode` remains the execution runtime.

## Effective Runtime Chain

- Orchestrator default coder provider: `opencode`
- Orchestrator default coder model: `qwen3-coder-plus-2025-07-22`
- Worker-coder default provider: `opencode`
- Worker-coder default model: `qwen3-coder-plus-2025-07-22`
- Execution command shape: `opencode run "<task_prompt>" --model qwen3-coder-plus-2025-07-22`

## Required Environment Variables

For `worker-coder`:

- `OPENCODE_BIN`
- `CODER_PROVIDER_DEFAULT=opencode`
- `CODER_MODEL_DEFAULT=qwen3-coder-plus-2025-07-22`
- `QWEN_API_KEY`
- `QWEN_BASE_URL`
- `DASH_SCOPE_API_KEY`
- `DASH_SCOPE_BASE_URL`

Notes:

- Nexus passes the model name to `opencode`; the internal provider resolution inside `opencode` is still owned by the OpenCode runtime.
- This repository no longer treats `qwen` as a standalone execution provider inside `worker-coder`.

## Verification Target

When the stack is running correctly, coding execution logs should show:

- `provider_used=opencode`
- `model_used=qwen3-coder-plus-2025-07-22`
- `command_used` containing `opencode run ... --model qwen3-coder-plus-2025-07-22`
