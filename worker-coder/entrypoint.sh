#!/bin/sh
set -eu

# Expose shared contracts at /shared so worker.js can import from ../shared/
if [ -d /workspace/shared ] && [ ! -e /shared ]; then
  ln -snf /workspace/shared /shared
fi

# Resolve provider API key placeholders into the runtime OpenCode config.
if [ -f /app/opencode.json.tpl ]; then
  mkdir -p /root/.config/opencode /root/.config/opencode/plugins
  if [ -d /workspace/external/vendor/superpowers ]; then
    ln -snf /workspace/external/vendor/superpowers /root/.config/opencode/superpowers
  fi

  if [ -z "${DASHSCOPE_API_KEY:-}" ]; then
    export DASHSCOPE_API_KEY="${DASH_SCOPE_API_KEY:-${QWEN_API_KEY:-${ALIBABA_CODING_PLAN_API_KEY:-}}}"
  fi

  if [ -d /workspace/external/vendor/superpowers ]; then
    export OPENCODE_PLUGIN_JSON='[
    "superpowers@git+https://github.com/obra/superpowers.git"
  ]'
  else
    export OPENCODE_PLUGIN_JSON='[]'
  fi

  envsubst '${MINIMAX_API_KEY} ${DASHSCOPE_API_KEY} ${DASH_SCOPE_API_KEY} ${QWEN_API_KEY} ${ALIBABA_CODING_PLAN_API_KEY} ${OPENCODE_PLUGIN_JSON}' \
    < /app/opencode.json.tpl \
    > /app/opencode.json
  mkdir -p /root/.config/opencode
  cp /app/opencode.json /root/.config/opencode/opencode.json
fi

exec node worker.js
