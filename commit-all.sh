#!/usr/bin/env bash

set -e

message="${1:-Automatic upload}"

git pull --rebase --autostash
git add --all

if git diff --cached --quiet; then
    printf '%s\n' 'No changes to commit.'
else
    git commit -m "$message"
fi

git push
