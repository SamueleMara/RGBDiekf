#!/usr/bin/env bash
set -euo pipefail
REPO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ "${1:-}" == inspect ]]; then
    shift
    exec bash "$REPO_DIR/scripts/run_full_scene.sh" --validate-only "$@"
fi
if [[ "${1:-}" == run ]]; then shift; fi
exec bash "$REPO_DIR/scripts/run_full_scene.sh" "$@"
