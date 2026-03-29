#!/bin/sh
set -eu

# Resolve provider API key placeholders into the runtime OpenCode config.
if [ -f /app/opencode.json.tpl ]; then
  envsubst '${MINIMAX_API_KEY} ${DASH_SCOPE_API_KEY} ${ALIBABA_CODING_PLAN_API_KEY}' \
    < /app/opencode.json.tpl \
    > /app/opencode.json
  mkdir -p /root/.config/opencode
  cp /app/opencode.json /root/.config/opencode/opencode.json
fi

exec node worker.js
