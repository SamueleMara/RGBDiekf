#!/usr/bin/env python3
"""Run the joint K = 1 runner (unchanged) on a synthetic masked RGB-D fixture (tests/svgd_fixtures.SyntheticDataset), inference only.
The fixture's pose / membership (its GT) are never passed to the estimator; afterwards the fixture PARAMETERS (not code) are written to
OUT/synthetic_fixture.json for the separate post-hoc oracle (tools/audit_svgd_joint_k1_oracles.py --synthetic).

    python -B tools/run_svgd_joint_k1_synthetic.py --motion translate --frames 10 --output OUT [--uniform-colour] [--config CFG]
"""
import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--motion', choices=['translate', 'rotate', 'none'], required=True); ap.add_argument('--frames', type=int, default=10)
    ap.add_argument('--speed', type=float, default=.15); ap.add_argument('--depth-noise', type=float, default=5e-4); ap.add_argument('--rgb-noise', type=float, default=0.)
    ap.add_argument('--n-cams', type=int, default=3); ap.add_argument('--seed', type=int, default=5); ap.add_argument('--uniform-colour', action='store_true')
    ap.add_argument('--config', type=Path, default=REPO/'configs/svgd_joint_k1.json'); ap.add_argument('--output', type=Path, required=True); ap.add_argument('--device', default='cuda')
    a = ap.parse_args(argv)
    from tests.svgd_fixtures import SyntheticDataset
    from estimator.svgd_k1_run import run
    spec = dict(K=a.frames, motion=a.motion, speed=a.speed, depth_noise=a.depth_noise, rgb_noise=a.rgb_noise, n_cams=a.n_cams, seed=a.seed,
                plate_colour=[.62, .35, .2] if a.uniform_colour else None)
    args = SimpleNamespace(scene='/synthetic', output=a.output, config=a.config, num_moving_parts=1, device=a.device, frame_limit=None, fps=None,
                           dt_seconds=None, timestamps_file=None, noise_config=None, sensor_noise='measured')
    code = run(args, data=SyntheticDataset(**spec))
    (a.output/'synthetic_fixture.json').write_text(json.dumps(spec, indent=1))
    return code


if __name__ == '__main__':
    sys.exit(main())
