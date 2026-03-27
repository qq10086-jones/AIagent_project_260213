#!/bin/sh
# Resolve ${VAR} placeholders in opencode.json at container startup.
# This keeps secrets out of the committed config file and supports both
# local-LLM and cloud-LLM environments via the same template.
if [ -f /app/opencode.json.tpl ]; then
  envsubst '${MINIMAX_API_KEY} ${DASH_SCOPE_API_KEY} ${ALIBABA_CODING_PLAN_API_KEY}' < /app/opencode.json.tpl > /app/opencode.json
  # Also write to user-level config so opencode finds it regardless of cwd
  mkdir -p /root/.config/opencode
  cp /app/opencode.json /root/.config/opencode/opencode.json
fi
exec node worker.js
