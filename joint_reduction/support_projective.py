"""Support completion from measured projective RGB-D consistency (contract section 6), downstream of an accepted joint.

For every free-filtered endpoint point i outside the rigid core and every non-anchor frame k, two hypotheses place it:
    static   x_s = p_i                       moving   x_m = T_J(q_k) p_i     (accepted joint, q from the q filter if run)
Each is projected into every camera and the ACTUAL frame-k RGB-D is sampled (projective_sensor, same validity rules).
Per camera and hypothesis, with Z the hypothesis depth and D the measured depth, sigma^2 = R_D + dZ/dx Sigma_p dZ/dx^T:
    D < Z - n sigma      measured surface in front: hidden OR wrong -> no evidence either way (excluded)
    otherwise            usable; cost c = r^T R^-1 r of the 4-channel innovation (a surface seen BEHIND the hypothesis
                         position, D > Z + n sigma, contradicts it: no occluder can explain empty space, so it counts)
A frame contributes d_k = sum over cameras usable under BOTH hypotheses of (c_static - c_moving) / (number of such
cameras); a frame with no common camera contributes nothing. Labels from frame-level one-sided tests on the mean of
d_k (conservative max(Newey-West, AR(1)) long-run variance around a linear trend, HLN, t(n-1); predictive_tests),
Benjamini-Hochberg over points at the declared alpha per family:
    MOVING_JOINT_COMPATIBLE   moving significantly more consistent than static on measured frames
    STATIC_COMPATIBLE         static significantly more consistent
    AMBIGUOUS                 neither (near-axis points, frames where q ~ 0, textureless/duplicate geometry ...)
    OCCLUDED_OR_UNOBSERVED    fewer than 3 frames with a common usable camera
    MOVING_CORE               the rigid core (unchanged)
No Gibbs energy, no point-splat flips, no ground truth; a label needs measured-frame evidence on visible frames.
"""
import json
import math
import time
from pathlib import Path
import numpy as np
import torch
from scipy import stats
from estimator.projective_camera import CameraRig
from estimator.projective_sensor import MeasuredFrame, SensorNoise, sample, CODE
from joint_reduction.joint_projective import joint_transform
from utils.io import json_write, serial


def hac_mean_test(D, valid):
    """Vectorized frame-level t statistics of the mean of each column of D [K,N] over its valid frames (in time order),
    with the conservative long-run variance of joint_reduction/predictive_tests.conservative_long_run_variance.
    Returns (n [N], mean [N], t_hln [N])."""
    K, N = D.shape
    order = np.argsort(~valid, axis=0, kind='stable')                 # valid frames first, time order kept
    X = np.take_along_axis(np.where(valid, D, 0.), order, 0); M = np.take_along_axis(valid, order, 0)
    n = M.sum(0)
    nn = np.maximum(n, 1)
    mean = X.sum(0)/nn
    t = np.where(M, np.arange(K)[:, None]-(n[None]-1)/2, 0.)
    dev = np.where(M, X-mean, 0.)
    beta = (t*dev).sum(0)/np.maximum((t*t).sum(0), 1e-300)
    e = np.where(M, dev-beta*t, 0.)
    g0 = (e*e).sum(0)/nn
    lag = np.floor(4*(nn/100.)**(2/9)).astype(int)
    nw = g0.copy()
    for l in range(1, int(lag.max())+1 if len(lag) else 1):
        gl = (e[l:]*e[:-l]).sum(0)/nn
        nw += np.where((l <= lag) & (l <= n-1), 2*(1-l/(lag+1))*gl, 0.)
    r = np.where(g0 > 0, (e[1:]*e[:-1]).sum(0)/np.maximum((e*e).sum(0), 1e-300), 0.)
    r = np.clip(r+(2+4*r)/nn, -.97, .97)
    s2 = np.maximum(np.maximum(nw, 0.), g0/(1-r)**2)
    with np.errstate(divide='ignore', invalid='ignore'):
        dm = np.where(s2 > 0, mean/np.sqrt(s2/nn), np.sign(mean)*np.inf)
    return n, mean, dm*np.sqrt(np.maximum(nn-1, 0)/nn)


def benjamini_hochberg(p, alpha):
    m = len(p); passed = np.zeros(m, bool)
    if m == 0: return passed
    o = np.argsort(p); ok = p[o] <= alpha*np.arange(1, m+1)/m
    if ok.any(): passed[o[:np.flatnonzero(ok).max()+1]] = True
    return passed


def _costs(rig, frame, noise, policy, x, rgb0, sig_p, chunk=400_000):
    """Per (camera, point) cost r^T R^-1 r and usability (sample valid and measured surface not in front)."""
    C, N = rig.count, len(x)
    cost = torch.full((C, N), float('nan'), dtype=torch.float64, device=x.device)
    for c in range(C):
        for a in range(0, N, chunk):
            xs = x[a:a+chunk]; cam = torch.full((len(xs),), c, dtype=torch.long, device=x.device)
            uv, Z, Ju, Jz = rig.project_rows(xs, cam)
            ok = (Z > rig.near[c]) & torch.isfinite(Z)
            s = sample(frame, cam, uv, noise, policy, rig.K, torch.float64)
            ok &= s.reason == CODE['VALID']
            var_z = s.var_depth+torch.einsum('mi,mij,mj->m', Jz.double(), sig_p[a:a+chunk], Jz.double())
            hidden = (Z-s.depth) > policy['visibility_n_sigma']*torch.sqrt(var_z)
            ok &= ~hidden
            r = torch.cat([rgb0[a:a+chunk]-s.rgb, (Z-s.depth)[:, None]], 1)
            var = torch.cat([s.var_rgb, var_z[:, None]], 1)
            cost[c, a:a+chunk] = torch.where(ok, (r*r/var).sum(1), torch.full_like(Z, float('nan'), dtype=torch.float64))
    return cost


def run_support(dataset, config, policy, free_dir, rigid_dir, joint_result, root, alpha, device):
    root = Path(root); root.mkdir(parents=True, exist_ok=True)
    kind = joint_result.get('status')
    if kind not in ('PRISMATIC', 'REVOLUTE'):
        st = {'status': 'SKIPPED_NO_ACCEPTED_JOINT', 'joint_status': kind}; json_write(root/'result.json', st); return st
    t0 = time.perf_counter()
    can = np.load(Path(free_dir)/'canonical.npz'); core = np.load(Path(rigid_dir)/'core_point_ids.npy')
    anchor = int(can['canonical_frame'])
    cand = joint_result['candidates'][kind]
    q = np.asarray(joint_result['q_filter']['q'] if 'q_filter' in joint_result else cand['q'], float)
    N, K = len(can['canonical_xyz']), len(dataset.times)
    test = np.setdiff1d(np.arange(N), core)
    dev = torch.device(device)
    rig = CameraRig(dataset.cameras, device); noise = SensorNoise.from_config(config, policy, device)
    p = torch.as_tensor(can['canonical_xyz'][test], dtype=torch.float64, device=dev)
    rgb0 = torch.as_tensor(can['canonical_rgb'][test], dtype=torch.float64, device=dev)
    sig = torch.as_tensor(can['canonical_source_covariance'][test], dtype=torch.float64, device=dev)
    D = np.zeros((K, len(test))); valid = np.zeros((K, len(test)), bool)
    usable_s = np.zeros(len(test), int); usable_m = np.zeros(len(test), int)
    for k in range(K):
        if k == anchor: continue
        frame = MeasuredFrame(dataset, k, device)
        T = torch.as_tensor(joint_transform(kind, cand['axis'], cand['axis_point'], float(q[k])), dtype=torch.float64, device=dev)
        cs = _costs(rig, frame, noise, policy, p, rgb0, sig)
        R = T[:3, :3]; xm = p@R.T+T[:3, 3]
        cm = _costs(rig, frame, noise, policy, xm, rgb0, R@sig@R.T)
        both = torch.isfinite(cs) & torch.isfinite(cm)
        nb = both.sum(0)
        d = torch.where(both, cs-cm, torch.zeros_like(cs)).sum(0)/nb.clamp_min(1)
        D[k] = d.cpu().numpy(); valid[k] = (nb > 0).cpu().numpy()
        usable_s += torch.isfinite(cs).any(0).cpu().numpy(); usable_m += torch.isfinite(cm).any(0).cpu().numpy()
    n, mean, t = hac_mean_test(D, valid)
    enough = n >= 3
    df = np.maximum(n-1, 1).astype(float)
    tt = np.nan_to_num(t, nan=0.)                                    # nan only for zero-variance zero-mean series
    tdist_sf = np.where(enough, stats.t.sf(tt, df), 1.); tdist_cdf = np.where(enough, stats.t.cdf(tt, df), 1.)
    mov = np.zeros(len(test), bool); sta = np.zeros(len(test), bool)
    mov[enough] = benjamini_hochberg(tdist_sf[enough], alpha); sta[enough] = benjamini_hochberg(tdist_cdf[enough], alpha)
    labels = np.full(N, 'AMBIGUOUS', dtype=object)
    labels[core] = 'MOVING_CORE'
    lt = np.full(len(test), 'AMBIGUOUS', dtype=object)
    lt[~enough] = 'OCCLUDED_OR_UNOBSERVED'; lt[mov & ~sta] = 'MOVING_JOINT_COMPATIBLE'; lt[sta & ~mov] = 'STATIC_COMPATIBLE'
    labels[test] = lt
    np.savez_compressed(root/'support.npz', labels=labels.astype(str), tested_point_ids=test, frame_differential=D.astype(np.float32),
                        frame_valid=valid, n_frames=n, mean_differential=mean, t_hln=t, p_moving=tdist_sf, p_static=tdist_cdf)
    counts = {k: int(v) for k, v in zip(*np.unique(labels.astype(str), return_counts=True))}
    out = {'status': 'COMPLETE', 'joint_type': kind, 'alpha': alpha, 'counts': counts, 'tested_points': int(len(test)), 'core_points': int(len(core)),
           'q_source': 'q_filter' if 'q_filter' in joint_result else 'reduction', 'seconds': time.perf_counter()-t0,
           'rule': 'frame-level HAC one-sided tests of mean(cost_static - cost_moving) over cameras usable under both hypotheses; BH-FDR per family',
           'scope': 'free-filtered (possibly subsampled) endpoint points only; omitted full-cloud points are not labelled',
           'ground_truth_used': False}
    json_write(root/'result.json', serial(out))
    return out
