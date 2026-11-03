#!/usr/bin/env python3
"""POST-INFERENCE GT-ONLY cause analysis of a material-following run, frame 1 (never read by inference).

Replays frame 1 of a completed run from its own canonical catalogue, manifest and noise calibration (checked against the
published frame), then separates the causes of moving-point error:
  1. ORACLE BOUND: exact data at the true state, material likelihood linearized at the truth, same 6x6 prior:
     d_hat = (P^-1 + I_x)^-1 I_x d  (what any estimator of this likelihood could reach without nonlinearity);
  2. GT-STARTED IEKF: the original-resolution IEKF started at the true current state (conditional mean of the local prior),
     kept only where the acceptance comparison against the prediction accepts it, and also forced;
  3. OBJECTIVE PREFERENCE: total local objective (exact Gaussian conditioning of all local variables on the position) at
     the accepted vs the true position, on identities valid at both.

    python -B tools/diagnose_material_observability.py --run RESULTS/temporal_rgbd_X/USB_100109_material-following --output OUT.json
"""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.evaluate_temporal_local_pilot import truth_for
from utils.io import json_write, serial

DATA = Path('/media/samuelemara/Extreme SSD/Samuele/Backup_09_10_26/Sapien/output/preparation_8cams/partnet_subset_fixed_render_linear')
Q = [.25, .5, .75, .9, .99]


def q_mm(x):
    return (1e3*np.quantile(x, Q)).tolist() if len(x) else None


def setup(run, device):
    from scene.dataset import Dataset
    from scene.config import load_config
    from estimator.projective_camera import CameraRig
    from estimator.projective_sensor import MeasuredFrame, SensorNoise
    from estimator.temporal_material_run import MaterialFilter
    root = run/'free_forward'
    top = json.loads((run/'run_manifest.json').read_text())
    cal = json.loads((run/'noise_calibration.json').read_text())['result']
    cfg = load_config()
    cfg.color_covariance = np.diag(cal['rgb_variance'])
    cfg.depth_variance, cfg.depth_squared_variance = cal['depth_variance_m2'], cal['depth_squared_variance']
    scene = Path(top['dataset']['input'])
    data = Dataset(scene, 2)
    rig = CameraRig(data.cameras, device, torch.float64)
    noise = SensorNoise.from_config(cfg, top['policy'], device, torch.float64)
    man = json.loads((root/'run_manifest.json').read_text())
    can = np.load(root/'canonical.npz')
    opts = top['temporal']
    F = {k: MeasuredFrame(data, k, device, torch.float64) for k in (0, 1)}

    def new_filter():
        f = MaterialFilter(rig, noise, top['policy'], np.asarray(opts['warp_covariance']), np.asarray(man['q_c_m2_s3']), opts['lag'],
                           opts['local_chunk_bytes'], 2, F[0], can['original_source_ids'], can['canonical_xyz'])
        f.resampling, f.pyramid_levels = opts.get('material_resampling', 'none'), int(opts.get('material_pyramid_levels', 0))
        f.initialize(F[0], np.asarray(man['noise_and_prior']['pixel_covariance']), np.asarray(man['noise_and_prior']['velocity_covariance']))
        return f
    return root, scene, data, F, new_filter, top


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--run', type=Path, required=True, help='<batch>/<scene>_material-following')
    ap.add_argument('--data-root', type=Path, default=DATA)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    a = ap.parse_args()
    if a.output.exists():
        ap.error('output exists; never overwrite')
    import estimator.temporal_material_iekf as MI
    from estimator.temporal_local_covariance import chunk_plan
    dev = torch.device(a.device)
    root, scene, data, F, new_filter, top = setup(a.run, dev)
    dt = float(data.times[1]-data.times[0])
    tol = float(top['policy']['iekf_step_tolerance'])
    truth = truth_for(root, a.data_root/scene.name)
    mv = torch.as_tensor(truth['moving'], device=dev)
    x0 = torch.as_tensor(truth['X'][0], device=dev)
    xg = torch.as_tensor(truth['X'][1], device=dev)
    published = np.load(root/'posterior_xyz.npy', mmap_mode='r')[1]
    out = {'ground_truth_namespace': 'POSTHOC_GT_DIAGNOSTIC_ONLY', 'run': str(a.run), 'frame': 1, 'labels': 'NN part-cloud labels (truth_for)'}

    filt = new_filter()
    st = filt.step(F, 1, 1, dt, 10, tol)
    up = st['update']
    out['replay_max_abs_vs_published_m'] = float(np.abs(st['s'][:, :3].cpu().numpy()-np.asarray(published)).max())
    xf = st['s'][:, :3].clone()

    def conditional(x, name, zname):
        up.P[name] = up.mu.clone()
        up.P[zname] = torch.zeros_like(up.mu)
        up.ex.v[name] = torch.zeros_like(up.ex.v[name])
        up.ex.v[zname] = torch.zeros_like(up.ex.v[zname])
        L = up.prior.layout
        for pts in chunk_plan(torch.arange(up.N, device=dev), L.dims, up.budget):
            mu, Sig = up.prior.gather(pts)
            n = mu.shape[1]
            w = torch.linalg.solve(Sig[:, :3, :3], (x[pts]-mu[:, :3]).unsqueeze(-1))
            aa = mu+(Sig[:, :, :3]@w).squeeze(-1)
            z = torch.zeros_like(mu)
            z[:, :3] = w.squeeze(-1)
            vm = torch.arange(n, device=dev)[None, :] < L.dims[pts][:, None]
            idx = (up.prior.moff[pts][:, None]+torch.arange(n, device=dev))[vm]
            up.P[name][idx] = aa[vm]
            up.P[zname][idx] = z[vm]
    # 3. objective preference
    res = {}
    for tag, x in (('accepted', xf), ('truth', xg)):
        conditional(x, 'trial', 'ztrial')
        lr = up.linearize('trial', register=False)
        c, v = up.identity_costs(lr)
        res[tag] = (c, v, up.prior_cost('trial', 'ztrial'), lr)
    common = res['accepted'][1] & res['truth'][1]
    Ja = res['accepted'][2]+(res['accepted'][0]*common).sum(0)
    Jt = res['truth'][2]+(res['truth'][0]*common).sum(0)
    err = (xf-xg).norm(dim=1)
    g = mv & common.any(0)
    out['objective_preference'] = {grp: {'points': int(h.sum()), 'truth_lower_objective_fraction': float((Jt[h] < Ja[h]).double().mean()) if bool(h.any()) else None}
                                   for grp, h in (('moving_error_gt_8mm', g & (err > .008)), ('moving_error_3_8mm', g & (err >= .003) & (err <= .008)),
                                                  ('moving_error_lt_3mm', g & (err < .003)))}
    # 1. oracle bound at the truth
    lt = res['truth'][3]
    R = torch.diag_embed(lt.r_eff+lt.rows.resampling_variance)+up.warp
    Hx = lt.rows.H_t[:, :, :3]
    info = torch.zeros((up.N, 3, 3), dtype=torch.float64, device=dev)
    info.index_add_(0, lt.rows.point, Hx.transpose(1, 2)@torch.linalg.solve(R, Hx))
    P = st['prior6'][:, :3, :3]
    d = xg-st['mean'][:, :3]
    dhat = torch.linalg.solve(torch.linalg.inv(P)+info, (info@d.unsqueeze(-1))).squeeze(-1)
    oerr = (dhat-d).norm(dim=1)
    obs = torch.zeros_like(mv)
    obs[lt.rows.point] = True
    h = mv & obs
    ev = torch.linalg.eigvalsh(info[h].cpu())
    rank = (ev > 1e-12*ev[:, 2:3]).sum(1)
    out['oracle'] = {'moving_points_with_rows_at_truth': int(h.sum()), 'quantiles': Q,
                     'prediction_error_mm': q_mm(d.norm(dim=1)[h].cpu().numpy()), 'oracle_bound_error_mm': q_mm(oerr[h].cpu().numpy()),
                     'actual_error_mm_same_points': q_mm(err[h].cpu().numpy()),
                     'information_rank_counts': {int(r): int((rank == r).sum()) for r in range(4)},
                     'prior_position_std_mm_median': (1e3*torch.linalg.eigvalsh(P[h].cpu()).clamp_min(0).sqrt().median(0).values).tolist()}
    # 2. GT-started IEKF (kept by the acceptance comparison / forced)
    out['gt_started'] = {}
    for forced in (False, True):
        rec = {}

        def start(self, lr, forced=forced, rec=rec):
            pts = mv.clone()
            s6 = torch.cat((xg, (xg-x0)/dt), 1)
            self.coarse_report = {'diagnostic': 'GT-started IEKF'}
            base = self._state(lr, 'it', 'z')
            self._condition_current(pts, s6)
            lr1 = self.linearize('it')
            joint = self._state(lr1, 'it', 'z')
            good, why, _, _ = self.compare(base, joint, joint['prior'])
            rec['kept_by_acceptance'] = int((good & pts).sum())
            rec['rejected_by_acceptance'] = {k: int((v & pts).sum()) for k, v in why.items()}
            if not forced:
                rev = pts & ~good
                if bool(rev.any()):
                    self._reset_to_prior(rev)
                    lr1 = self.linearize('it')
                self.coarse_started = pts & good
            else:
                self.coarse_started = pts.clone()
            return lr1
        orig = MI.MaterialFrameUpdate._pyramid_start
        MI.MaterialFrameUpdate._pyramid_start = start
        try:
            f2 = new_filter()
            f2._pyramid = lambda frames, k: ('gt', [1])
            s2 = f2.step(F, 1, 1, dt, 10, tol)
        finally:
            MI.MaterialFrameUpdate._pyramid_start = orig
        e2 = (s2['s'][:, :3]-xg).norm(dim=1)
        rec['final_moving_error_mm'] = q_mm(e2[mv].cpu().numpy())
        rec['static_error_mm_p99_p999'] = (1e3*np.quantile(e2[~mv].cpu().numpy(), [.99, .999])).tolist()
        out['gt_started']['forced' if forced else 'acceptance_checked'] = rec
    out['actual_moving_error_mm_all'] = q_mm(err[mv].cpu().numpy())
    json_write(a.output, serial(out))
    print(json.dumps(out, indent=1)[:4000])


if __name__ == '__main__':
    raise SystemExit(main())
