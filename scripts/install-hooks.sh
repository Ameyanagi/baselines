#!/usr/bin/env sh
set -eu

git config core.hooksPath .githooks
chmod +x .githooks/pre-commit
printf '%s\n' "Installed git hooks from .githooks"
