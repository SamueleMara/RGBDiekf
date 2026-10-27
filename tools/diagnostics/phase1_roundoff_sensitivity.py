#!/usr/bin/env python3
"""Is the multi-frame Phase-1 trajectory sensitive to round-off independently of B1? Runs the production Phase 1 (first
anchor, --frames frames) three times into a scratch directory: OFF; OFF with the predicted state multiplied by
(1 + 1e-15 * N(0,1)) every frame (a round-off-sized perturbation, no algorithm change); ON (small_contractions).
Reports per-frame position differences of each pair. Writes only scratch outputs + reports/diagnostics JSON."""
import sys, json, itertools
from pathlib import Path
REPO = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(REPO))
import numpy as np, torch
from pointwise_rgbd.dataset import Dataset
from pointwise_rgbd.config import load_config
from pointwise_rgbd import image_phase1
from pointwise_rgbd.image_iekf import IEKFConfig, frame_step
from pointwise_rgbd.rgbd_point_renderer import RenderPolicy
from pointwise_rgbd.image_observation import ObservationPolicy
from rgbd_common.backends.torch_cuda import Backend
scene, scratch, frames = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3])
config = load_config(); data = Dataset(scene, frames, None, fps=59); backend = Backend('cuda')
policy = RenderPolicy(depth_variance=config.depth_variance, depth_squared_variance=config.depth_squared_variance)
catalog = image_phase1.endpoint_catalogue(data, 'first', config, IEKFConfig())
gen = torch.Generator(device='cpu').manual_seed(0)
def perturbed(s_minus, *a):
    noise = torch.randn(s_minus.shape, generator=gen, dtype=s_minus.dtype).to(s_minus.device)
    return frame_step(s_minus*(1+1e-15*noise), *a)
states = {}
for name, cfg, step in (('off', IEKFConfig(), frame_step), ('off_perturbed_1e-15', IEKFConfig(), perturbed), ('on', IEKFConfig(small_contractions=True), frame_step)):
    torch.manual_seed(0); image_phase1.frame_step = step
    _, arrays, _ = image_phase1.run(data, config, backend, scratch/name, 'first', cfg, policy, ObservationPolicy(), 1.5, catalog=catalog, save_every=0)
    states[name] = np.array(arrays['state'])
image_phase1.frame_step = frame_step
out = {'frames': frames, 'pairs': {}}
for a, b in itertools.combinations(states, 2):
    d = np.linalg.norm(states[a][:, :, :3]-states[b][:, :, :3], axis=2)
    out['pairs'][f'{a} vs {b}'] = [{'frame': k, 'median_mm': 1e3*float(np.median(d[k])), 'p90_mm': 1e3*float(np.quantile(d[k], .9)),
                                    'max_mm': 1e3*float(d[k].max()), 'points_over_1cm': int((d[k] > .01).sum())} for k in range(frames)]
    print(a, 'vs', b, [(r['frame'], round(r['median_mm'], 3), round(r['p90_mm'], 2), round(r['max_mm'], 1)) for r in out['pairs'][f'{a} vs {b}']], flush=True)
(REPO/'reports/diagnostics/phase1_roundoff_sensitivity.json').write_text(json.dumps(out, indent=1))
