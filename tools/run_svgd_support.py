#!/usr/bin/env python3
"""Opt-in SVGD-assisted moving-support Stage 1 (CODEX_RGBDiekf_SVGD_Sequence_Moving_Support_CUDA_THEORY_PARITY_2026-11-03.md).

    python -B tools/run_svgd_support.py --scene SCENE --config configs/svgd_support.json --device cuda --dry-run
    python -B tools/run_svgd_support.py --scene SCENE --config configs/svgd_support.json --device cuda --frame-limit 3 --debug full --output NEW_DIR
    python -B tools/run_svgd_support.py --scene SCENE --config configs/svgd_support_experimental.json --svgd-theory-fidelity experimental ...

Strict (default) refuses every undeclared law / unauthorized deviation and stops at the first blocked stage (exit 3);
experimental runs PROPOSED deviations with quarantined, never-probabilistic outputs. Exit: 0 ok/stopped, 3 BLOCKED,
4 CUDA_OOM (journal written), 5 failed. A --frame-limit run is a PREFIX_DIAGNOSTIC, never a whole-sequence estimate.
The legacy train.py CLI and stages are untouched.
"""
import argparse
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from estimator.svgd_contract import STAGES

RESULTS_ROOT = Path('/media/samuelemara/Extreme SSD/Samuele/results_11_10_26/RGBDekf')     # same convention as train.py


def parse(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--scene', type=Path, required=True)
    ap.add_argument('--output', type=Path, help='new empty directory outside the repository (default RESULTS_ROOT/svgd_<scene>_<YYYYmmdd_HHMMSS>)')
    ap.add_argument('--config', type=Path, default=REPO/'configs/svgd_support.json')
    ap.add_argument('--svgd-theory-fidelity', choices=['strict', 'experimental'], default='strict')
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--frame-limit', type=int)
    ap.add_argument('--fps', type=float); ap.add_argument('--dt-seconds', type=float); ap.add_argument('--timestamps-file', type=Path)
    ap.add_argument('--noise-config', type=Path)
    ap.add_argument('--sensor-noise', choices=['measured', 'config'], default='measured',
                    help='measured: scene/noise_calibration temporal MAD sensor estimate (no cross-camera mismatch); config: approved noise file')
    ap.add_argument('--debug', choices=['none', 'light', 'full'], default='light')
    ap.add_argument('--display-cap', type=int, default=60000, help='display-only points per cloud in the viewer (inference always uses every sample)')
    ap.add_argument('--debug-point-ids', type=int, nargs='*', help='original canonical IDs to trace (default: 8 evenly spaced IDs)')
    ap.add_argument('--svgd-workspace-mib', type=float, default=512., help='operational memory budget of one operator tile (never a bandwidth)')
    ap.add_argument('--svgd-kernel-backend', choices=['torch', 'triton'], default='torch')
    ap.add_argument('--stop-after-step', choices=STAGES)
    ap.add_argument('--save-stein-diagnostics', action='store_true',
                    help='save outer-1 Phi / score / gradients / velocity lift to svgd_support/stein_diagnostics.npz for tools/audit_svgd_stein_rejection.py')
    ap.add_argument('--dry-run', action='store_true', help='validate config and inventory only; nothing is written')
    a = ap.parse_args(argv)
    if a.output is None and not a.dry_run: a.output = RESULTS_ROOT/f"svgd_{a.scene.resolve().name}_{time.strftime('%Y%m%d_%H%M%S')}"
    if a.frame_limit is not None and a.frame_limit < 1: ap.error('--frame-limit must be >= 1')
    if a.display_cap < 1: ap.error('--display-cap must be positive')
    return a


def main(argv=None):
    a = parse(argv)
    from estimator.svgd_support_run import run
    code = run(a)
    if not a.dry_run: print(f'[svgd] exit {code}; output {a.output}', flush=True)
    return code


if __name__ == '__main__':
    sys.exit(main())
