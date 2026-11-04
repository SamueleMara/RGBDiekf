#!/usr/bin/env python3
"""Joint K = 1 moving-support + shared rigid motion with SVGD over motion hypotheses (CODEX_RGBDiekf_SVGD_JOINT_RIGID_SUPPORT_K1_2026-11-03.md).

    python -B tools/run_svgd_joint_k1.py --scene SCENE --config configs/svgd_joint_k1_experimental.json --num-moving-parts 1 \
        --device cuda --frame-limit 10 --output NEW_DIR --debug full

--num-moving-parts other than 1 -> NOT_IMPLEMENTED_K_GT_1. --frame-limit -> PREFIX_DIAGNOSTIC. Legacy CLIs are untouched.
"""
import argparse
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
RESULTS_ROOT = Path('/media/samuelemara/Extreme SSD/Samuele/results_11_10_26/RGBDekf')


def parse(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--scene', type=Path, required=True); ap.add_argument('--output', type=Path)
    ap.add_argument('--config', type=Path, default=REPO/'configs/svgd_joint_k1.json'); ap.add_argument('--num-moving-parts', type=int, required=True)
    ap.add_argument('--device', default='cuda'); ap.add_argument('--frame-limit', type=int)
    ap.add_argument('--fps', type=float); ap.add_argument('--dt-seconds', type=float); ap.add_argument('--timestamps-file', type=Path)
    ap.add_argument('--noise-config', type=Path); ap.add_argument('--sensor-noise', choices=['measured', 'config'], default='measured')
    ap.add_argument('--debug', choices=['none', 'light', 'full'], default='light'); ap.add_argument('--display-cap', type=int, default=60000)
    a = ap.parse_args(argv)
    if a.num_moving_parts != 1: ap.exit(2, 'NOT_IMPLEMENTED_K_GT_1: only --num-moving-parts 1 is implemented\n')
    if a.output is None: a.output = RESULTS_ROOT/f"svgd_joint_k1_{a.scene.resolve().name}_{time.strftime('%Y%m%d_%H%M%S')}"
    return a


def main(argv=None):
    a = parse(argv)
    from estimator.svgd_k1_run import run
    code = run(a)
    if a.debug != 'none':
        try:
            from debug.svgd_joint_k1_debug import write_debug
            write_debug(a.output, a.debug, display_cap=a.display_cap)
        except ImportError: print('[k1] viewer module not available: debug pages not written', flush=True)
    return code


if __name__ == '__main__':
    sys.exit(main())
