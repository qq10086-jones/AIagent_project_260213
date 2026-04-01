#!/bin/sh
set -eu

# Expose shared contracts at /shared so worker.js can import from ../shared/
if [ -d /workspace/shared ] && [ ! -e /shared ]; then
  ln -snf /workspace/shared /shared
fi

# Resolve provider API key placeholders into the runtime OpenCode config.
if [ -f /app/opencode.json.tpl ]; then
  mkdir -p /root/.config/opencode /root/.config/opencode/plugins
  if [ -d /workspace/vendor/superpowers ] && [ -f /workspace/vendor/superpowers/.opencode/plugins/superpowers.js ]; then
    ln -snf /workspace/vendor/superpowers /root/.config/opencode/superpowers
    ln -snf /workspace/vendor/superpowers/.opencode/plugins/superpowers.js /root/.config/opencode/plugins/superpowers.js
  fi

  if [ -z "${DASHSCOPE_API_KEY:-}" ]; then
    export DASHSCOPE_API_KEY="${DASH_SCOPE_API_KEY:-${QWEN_API_KEY:-${ALIBABA_CODING_PLAN_API_KEY:-}}}"
  fi

  if [ -L /root/.config/opencode/plugins/superpowers.js ] || [ -f /root/.config/opencode/plugins/superpowers.js ]; then
    export OPENCODE_PLUGINS_JSON='[
    "/root/.config/opencode/plugins/superpowers.js"
  ]'
  else
    export OPENCODE_PLUGINS_JSON='[]'
  fi

  envsubst '${MINIMAX_API_KEY} ${DASHSCOPE_API_KEY} ${DASH_SCOPE_API_KEY} ${QWEN_API_KEY} ${ALIBABA_CODING_PLAN_API_KEY} ${OPENCODE_PLUGINS_JSON}' \
    < /app/opencode.json.tpl \
    > /app/opencode.json
  mkdir -p /root/.config/opencode
  cp /app/opencode.json /root/.config/opencode/opencode.json
fi

exec node worker.js
