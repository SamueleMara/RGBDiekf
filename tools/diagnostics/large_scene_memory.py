#!/usr/bin/env python3
"""Where a larger scene runs out of GPU memory under a per-process cap: render one frame (counters, peak), then the frame
update; on OOM report the requested allocation and the failing source line. Read-only; no outputs except the JSON."""
import sys, json, traceback
from pathlib import Path
REPO = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(REPO))
import numpy as np, torch
scene, frac = Path(sys.argv[1]), float(sys.argv[2]); torch.cuda.set_per_process_memory_fraction(frac); dev = 'cuda'
from pointwise_rgbd.dataset import Dataset
from pointwise_rgbd.config import load_config
from pointwise_rgbd.image_phase1 import endpoint_catalogue
from pointwise_rgbd.rgbd_point_renderer import footprints, CameraTensors, RenderPolicy, render
from pointwise_rgbd.supernodes import GroupBasis
from pointwise_rgbd.state6 import anchor_posterior
from pointwise_rgbd.process6 import predict
from pointwise_rgbd.image_iekf import IEKFConfig, frame_step
from pointwise_rgbd.image_observation import levels, ObservationPolicy
config, cfg = load_config(), IEKFConfig(); data = Dataset(scene, 2, None, fps=59)
t = lambda a, dt=torch.float64: torch.as_tensor(np.asarray(a), dtype=dt, device=dev)
_, part, _ = endpoint_catalogue(data, 'first', config, cfg)
rho = t(footprints(part, data, 1.5)); G = GroupBasis(t(part.xyz), t(part.membership, torch.long), t(part.centroids))
policy = RenderPolicy(depth_variance=config.depth_variance, depth_squared_variance=config.depth_squared_variance)
cams = [CameraTensors(c, torch.device(dev)) for c in data.cameras]
st, p = anchor_posterior(part.xyz, part.source_covariance, config.velocity_covariance)
sm, Pm = predict(t(st), t(p), float(data.times[1]-data.times[0]), config)
out = {'scene': scene.name, 'memory_fraction': frac, 'cap_MB': frac*torch.cuda.get_device_properties(0).total_memory/2**20, 'points': len(sm), 'groups': G.count}
torch.cuda.reset_peak_memory_stats(); lin = render(sm[:, :3], t(part.rgb), rho, cams, policy)
out['render_only'] = {'contributions': sum(len(v.point) for v in lin.views), 'candidates': sum(v.stats['candidate_contributors'] for v in lin.views),
                      'peak_allocated_MB': torch.cuda.max_memory_allocated()/2**20, 'linearization_MB': sum(v.jac.numel()*8 for v in lin.views)/2**20}
del lin; torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
try:
    lv = levels([data.observation(1, c) for c in range(len(cams))], config, ObservationPolicy(), dev)
    frame_step(sm, Pm.clone(), lv, t(part.rgb), rho, policy, G, cfg); out['frame_step'] = 'completed'
except torch.cuda.OutOfMemoryError as e:
    tb = [f'{f.filename.split("RGBDekf/")[-1]}:{f.lineno} {f.name}: {f.line}' for f in traceback.extract_tb(e.__traceback__) if 'RGBDekf' in f.filename]
    out['frame_step'] = {'oom': str(e).split('GiB')[0]+'GiB', 'stack': tb[-4:]}
out['frame_step_peak_allocated_MB'] = torch.cuda.max_memory_allocated()/2**20
(REPO/f'reports/diagnostics/large_scene_memory_{scene.name}.json').write_text(json.dumps(out, indent=1)); print(json.dumps(out, indent=1))
