#!/usr/bin/env sh
set -eu

cd "$(dirname "$0")/.."
npm install
node impl/be_changes/server.js
