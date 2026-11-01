#!/usr/bin/env bash
set -euo pipefail
REPO="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${RGBD_PYTHON:-$HOME/miniconda3/envs/rigeuler3dgs/bin/python}"
exec "$PYTHON" -B "$REPO/tools/run_temporal_rgbd_critical.py" "$@"
