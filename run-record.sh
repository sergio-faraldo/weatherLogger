#!/usr/bin/env bash

set -euo pipefail

script_directory="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
lock_file="/tmp/weatherlogger.lock"

if ! command -v flock >/dev/null 2>&1; then
    printf '%s\n' 'flock is required but was not found.' >&2
    exit 1
fi

if ! command -v timeout >/dev/null 2>&1; then
    printf '%s\n' 'timeout is required but was not found.' >&2
    exit 1
fi

uv_command="${UV_BIN:-/usr/local/bin/uv}"
if [[ ! -x "$uv_command" ]]; then
    uv_command="$(command -v uv || true)"
fi
if [[ -z "$uv_command" ]]; then
    printf '%s\n' 'uv is required but was not found.' >&2
    exit 1
fi

set +e
flock -n "$lock_file" timeout --signal=TERM --kill-after=5s 50s \
    "$uv_command" run --project "$script_directory" python "$script_directory/main.py" \
    --mode record --no-ui
status=$?
set -e

if [[ "$status" -eq 124 ]]; then
    printf '%s\n' 'Weather recording timed out after 50 seconds.' >&2
fi
exit "$status"