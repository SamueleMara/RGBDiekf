#!/usr/bin/env python3
"""Run the UNCHANGED production SVGD runner on a synthetic masked RGB-D fixture (tests/svgd_fixtures.SyntheticDataset), inference only.

    python -B tools/svgd_synthetic_run.py --motion translate --frames 10 --output OUT [--config configs/svgd_support_experimental.json]
    python -B tools/audit_svgd_falsification.py --run OUT --synthetic                  # separate, post-hoc GT audit

The fixture's pose (its GT) is never passed to the runner; after the run, the fixture PARAMETERS (not code) are written to
OUT/synthetic_fixture.json so that the separate audit can rebuild the same data and, post hoc only, its GT trajectory.
Sensor noise: measured by the same routine as for real data (--sensor-noise measured); the fixture's declared noise is its sensor.
"""
import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--motion', choices=['translate', 'rotate', 'none'], required=True); ap.add_argument('--frames', type=int, default=10)
    ap.add_argument('--speed', type=float, default=.15); ap.add_argument('--depth-noise', type=float, default=5e-4); ap.add_argument('--rgb-noise', type=float, default=0.)
    ap.add_argument('--n-cams', type=int, default=3); ap.add_argument('--seed', type=int, default=5)
    ap.add_argument('--config', type=Path, default=REPO/'configs/svgd_support_experimental.json'); ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--device', default='cuda'); ap.add_argument('--stop-after-step', default='STEIN_PROPOSAL')
    a = ap.parse_args(argv)
    from tests.svgd_fixtures import SyntheticDataset, runner_args
    from estimator.svgd_support_run import run
    spec = dict(K=a.frames, motion=a.motion, speed=a.speed, depth_noise=a.depth_noise, rgb_noise=a.rgb_noise, n_cams=a.n_cams, seed=a.seed)
    code = run(runner_args(a.output, str(a.config), mode='experimental', device=a.device, sensor_noise='measured', debug='light',
                           stop_after_step=a.stop_after_step, save_stein_diagnostics=True), data=SyntheticDataset(**spec))
    (a.output/'synthetic_fixture.json').write_text(json.dumps(spec, indent=1))
    print(f'[synthetic] exit {code}; fixture parameters -> {a.output}/synthetic_fixture.json', flush=True)
    return code


if __name__ == '__main__':
    sys.exit(main())
