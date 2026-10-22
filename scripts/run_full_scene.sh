#!/usr/bin/env bash
set -euo pipefail
REPO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
INPUT="${RGBDEKF_INPUT:-/media/samuelemara/Extreme SSD/Samuele/Backup_09_10_26/Sapien/output/real/fridge_plastic_20261016_163314}"
OUTPUT_ROOT="${RGBDEKF_OUTPUT_ROOT:-/media/samuelemara/Extreme SSD/Samuele/results_11_10_26/RGBDekf}"
mkdir -p -- "$OUTPUT_ROOT"
RUN_ROOT="$(mktemp -d "$OUTPUT_ROOT/rgbd_iekf.XXXXXXXX")"
mkdir -p -- "$RUN_ROOT/runtime/tmp" "$RUN_ROOT/runtime/cache"
mkdir -p -- "$RUN_ROOT/reports"
export TMPDIR="$RUN_ROOT/runtime/tmp" XDG_CACHE_HOME="$RUN_ROOT/runtime/cache"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OPENBLAS_NUM_THREADS=1
cd -- "$REPO_DIR"
CONDA="${CONDA_EXE:-/home/samuelemara/miniconda3/bin/conda}"
echo "Full-scene artifacts: $RUN_ROOT"
rc=0
"$CONDA" run --no-capture-output -n rigeuler3dgs python -B main_iekf.py \
    --input "$INPUT" --output "$RUN_ROOT/results" "$@" 2>&1 | tee "$RUN_ROOT/run.log" || rc=$?
if [[ -d "$RUN_ROOT/results/reports" ]]; then
    cp -a -- "$RUN_ROOT/results/reports/." "$RUN_ROOT/reports/"
fi
echo "Artifacts, diagnostics and log: $RUN_ROOT"
exit "$rc"
