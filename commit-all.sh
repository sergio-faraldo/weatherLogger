#!/usr/bin/env bash

set -e

message="${1:-Automatic upload}"

git add --all

if git diff --cached --quiet; then
    printf '%s\n' 'No changes to commit.'
    exit 0
fi

git commit -m "$message"
