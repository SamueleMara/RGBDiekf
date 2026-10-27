#!/usr/bin/env python3
"""ABLATION (not production): Phase 1 only on a real scene with a modified IEKFConfig, to diagnose a failing gate.
Example: --max-iters 30 (reference budget is 10). Ground truth is not read here."""
import argparse, json, os, sys
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
from pathlib import Path
REPO = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(REPO))
from pointwise_rgbd.dataset import Dataset
from pointwise_rgbd.config import load_config
from pointwise_rgbd.image_iekf import IEKFConfig
from pointwise_rgbd.rgbd_point_renderer import RenderPolicy
from pointwise_rgbd.image_observation import ObservationPolicy
from pointwise_rgbd import image_phase1
from rgbd_common.backends.torch_cuda import Backend

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument('--scene', type=Path, required=True); ap.add_argument('--output', type=Path, required=True); ap.add_argument('--fps', type=float, default=59)
ap.add_argument('--anchor', default='first'); ap.add_argument('--max-iters', type=int, default=10); ap.add_argument('--posterior', default='rigeuler_block_information')
a = ap.parse_args()
if a.output.exists() and any(a.output.iterdir()): ap.error('fresh output directory required')
data = Dataset(a.scene, None, None, fps=a.fps); config = load_config()
iekf = IEKFConfig(max_iters=a.max_iters, posterior=a.posterior)
(a.output).mkdir(parents=True, exist_ok=True); (a.output/'manifest.json').write_text(json.dumps({'ablation': True, 'iekf': iekf.manifest(), 'arguments': {k: str(v) for k, v in vars(a).items()}}, indent=1))
image_phase1.run(data, config, Backend('cuda'), a.output/a.anchor/'phase1', a.anchor, iekf,
                 RenderPolicy(depth_variance=config.depth_variance, depth_squared_variance=config.depth_squared_variance), ObservationPolicy())
