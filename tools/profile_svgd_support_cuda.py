#!/usr/bin/env python3
"""CUDA timing / memory profile of the exact SVGD Stage-1 operators (spec 9.2, Gate 8) with CPU float64 parity.

    python -B tools/profile_svgd_support_cuda.py --sizes 1000 10000 100000          # synthetic clouds (N = M)
    python -B tools/profile_svgd_support_cuda.py --run RUN [--sizes ...]            # + that run's performance.json, real canonical cloud

Per size: CUDA-event time and peak allocated/reserved bytes of target_score, stein_field, RFF mean/JVP/VJP and the exact
temporal solve, plus max relative CPU-vs-CUDA discrepancy on the first 512 queries (CPU dense reference). Exact: every
operator sums all N x M / N x N pairs; --svgd-workspace-mib only bounds tile memory. Writes JSON to stdout or --out.
"""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from estimator.svgd_target_score import target_score
from estimator.svgd_particle_operator import stein_field
from estimator.svgd_feature_measurement import RFFFeatures
from estimator.svgd_temporal import TemporalPrior


def timed(fn, *a, **k):
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record(); r = fn(*a, **k); e.record(); torch.cuda.synchronize()
    return r, {'seconds': s.elapsed_time(e)/1e3, 'peak_allocated_bytes': torch.cuda.max_memory_allocated(), 'peak_reserved_bytes': torch.cuda.max_memory_reserved()}


def rel(a, b): a, b = a.detach().cpu(), b.detach().cpu(); return float((a-b).abs().max()/(a.abs().max()+1e-300))


def profile(x, c, ws, frames=10):
    N = len(x); dev = 'cuda'; y = x+torch.randn_like(x)*1e-3; d = (c+torch.randn_like(c)*.01).clamp(0, 1)
    LX = torch.eye(3, dtype=torch.float64, device=dev).expand(N, 3, 3)*4e-6; LC = torch.eye(3, dtype=torch.float64, device=dev)*.003
    out = {'N': N, 'M': N}
    r, out['target_score'] = timed(target_score, x, c, y, d, LX, LC, workspace_bytes=ws)
    k = min(512, N); rc = target_score(x[:k].cpu(), c[:k].cpu(), y.cpu(), d.cpu(), LX.cpu(), LC.cpu())
    out['target_score']['cpu_cuda_max_rel'] = rel(rc.grad_xyz, r.grad_xyz[:k])
    H = torch.eye(3, dtype=torch.float64, device=dev)*1e-4
    sf, out['stein_field'] = timed(stein_field, x, c, r.grad_xyz, H, LC, workspace_bytes=ws)
    f = RFFFeatures(H.cpu(), LC.cpu(), 1024, 0, device=dev); u = torch.randn_like(x); w = torch.randn(1024, dtype=torch.float64, device=dev)
    _, out['rff_mean'] = timed(f.mean, x, c); _, out['rff_jvp'] = timed(f.jvp, x, c, u); _, out['rff_vjp'] = timed(f.vjp, x, c, w)
    tp = TemporalPrior(np.arange(frames)/30., np.eye(3)*.1, x, LX, np.eye(3)*.01)
    _, out[f'temporal_solve_{frames}_frames'] = timed(tp.solve, torch.randn(N, frames, 6, dtype=torch.float64, device=dev))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--run', type=Path); ap.add_argument('--sizes', type=int, nargs='*', default=[1000, 10000])
    ap.add_argument('--svgd-workspace-mib', type=float, default=512.); ap.add_argument('--out', type=Path)
    a = ap.parse_args()
    if not torch.cuda.is_available(): print(json.dumps({'status': 'CUDA_UNAVAILABLE'})); return 2
    torch.backends.cuda.matmul.allow_tf32 = False
    rep = {'device': torch.cuda.get_device_name(0), 'total_memory_bytes': torch.cuda.get_device_properties(0).total_memory, 'precision': 'float64, TF32 off', 'results': []}
    g = torch.Generator('cuda').manual_seed(0)
    if a.run:
        rep['run_performance'] = json.loads((a.run/'svgd_support/performance.json').read_text())
        can = np.load(a.run/'svgd_support/canonical.npz'); xr = torch.tensor(can['xyz0'], device='cuda'); cr = torch.tensor(can['rgb0'], device='cuda')
        rep['results'].append({'workload': 'run_canonical', **profile(xr, cr, int(a.svgd_workspace_mib*2**20))})
    for n in a.sizes:
        x = torch.randn(n, 3, dtype=torch.float64, device='cuda', generator=g)*.05; c = torch.rand(n, 3, dtype=torch.float64, device='cuda', generator=g)
        try: rep['results'].append({'workload': f'synthetic_{n}', **profile(x, c, int(a.svgd_workspace_mib*2**20))})
        except torch.cuda.OutOfMemoryError as e: rep['results'].append({'workload': f'synthetic_{n}', 'status': 'CUDA_OOM', 'error': str(e)[:300]})
        torch.cuda.empty_cache()
    txt = json.dumps(rep, indent=1)
    (a.out.write_text(txt) if a.out else print(txt))
    return 0


if __name__ == '__main__':
    sys.exit(main())
