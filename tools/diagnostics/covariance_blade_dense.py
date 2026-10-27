#!/usr/bin/env python3
"""Covariance diagnosis on a small version of the failing blade configuration (contract section 5).

A catalogue crop of ~N points around the moving/static interface of blade_103706 (all 8 cameras, real images) is
filtered with the production solver and the 'restricted_joseph_marginals' posterior (the method that diverged). At every
frame, with the SAME final linearization and the SAME gain (B from the method's own block prior):
  one_step  dense float64 Joseph as a SUM OF PSD TERMS from the method's block-diagonal prior: (I-KH) P (I-KH)^T + K R K^T
            -> isolates numerical error of the expanded marginal formula
  full      dense Joseph recursion with the full propagated covariance -> the true error covariance of the estimator
reported per frame: cond(B), min eigenvalues, max variances, block discrepancies. GT is not used.
"""
import argparse, json, sys, time
from pathlib import Path
REPO = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(REPO))
import numpy as np, torch
from pointwise_rgbd.dataset import Dataset
from pointwise_rgbd.config import load_config
from pointwise_rgbd.image_iekf import IEKFConfig, frame_step
from pointwise_rgbd.image_phase1 import endpoint_catalogue
from pointwise_rgbd.image_observation import levels, ObservationPolicy
from pointwise_rgbd.rgbd_point_renderer import RenderPolicy, footprints, CameraTensors, render
from pointwise_rgbd.supernodes import GroupBasis, build_supernodes
from pointwise_rgbd.state6 import anchor_posterior
from pointwise_rgbd.process6 import transition, process_covariance
from pointwise_rgbd.phase1_covariance import BlockCovariance, StructuredCovariance
from pointwise_rgbd.restricted_posterior import channel_weights

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument('--scene', type=Path, required=True); ap.add_argument('--out', type=Path, required=True)
ap.add_argument('--points', type=int, default=1000); ap.add_argument('--method', default='restricted_joseph_marginals'); ap.add_argument("--tag", default=""); ap.add_argument('--frames', type=int, default=40); ap.add_argument('--fps', type=float, default=59)
a = ap.parse_args()
dev = torch.device('cuda'); config = load_config(); iekf = IEKFConfig(posterior=a.method)
ds = Dataset(a.scene, a.frames+1, None, fps=a.fps)
k0, full, _ = endpoint_catalogue(ds, 'first', config, iekf)
rho_all = footprints(full, ds, 1.5)
# crop: the points nearest to the catalogue point whose x is median among those within the scene's middle (no GT used)
centre = np.median(full.xyz, 0)
keep = np.argsort(np.linalg.norm(full.xyz-centre, axis=1))[:a.points]
xyz, rgb, rho, cov0 = full.xyz[keep], full.rgb[keep], rho_all[keep], full.source_covariance[keep]
t = lambda x, d=torch.float64: torch.as_tensor(np.asarray(x), dtype=d, device=dev)
sn = build_supernodes(torch.as_tensor(xyz), target_size=64)
G = GroupBasis(t(xyz), sn.supernode_id.to(dev), sn.centroids.to(dev))
n, S = len(xyz), G.count
st, P0 = anchor_posterior(xyz, cov0, config.velocity_covariance)
s = t(st); cov = StructuredCovariance(t(P0), G) if a.method == 'restricted_joseph_structured' else BlockCovariance(t(P0), a.method)
Pfull = torch.block_diag(*t(P0)).cpu()
policy = RenderPolicy(depth_variance=config.depth_variance, depth_squared_variance=config.depth_squared_variance)
Gd = torch.zeros(6*n, 12*S, dtype=torch.float64); Gm = G.matrix().cpu(); mem = G.member.cpu()
for i in range(n): Gd[6*i:6*i+6, 12*mem[i]:12*mem[i]+12] = Gm[i]
null, _ = G.null_projector(); Pi = torch.block_diag(*null.cpu())
rows = []
for k in range(1, a.frames+1):
    clock = time.time()
    dt = float(ds.times[k]-ds.times[k-1])
    F, Q = transition(dt), process_covariance(dt, config.qc, config.qx)
    Fn, Qn = torch.block_diag(*[torch.tensor(F)]*n), torch.block_diag(*[torch.tensor(Q)]*n)
    s_minus = s@t(F).T; cov.predict(dt, False, config); Pfull = Fn@Pfull@Fn.T+Qn
    Pm = cov.marginals().clone()
    frames = [ds.observation(k, c) for c in range(len(ds.cameras))]
    lv = levels(frames, config, ObservationPolicy(), dev)
    s, P, observed, lin, terms, rep = frame_step(s_minus, cov, lv, t(rgb), t(rho), policy, G, iekf)
    # dense H^T W H and H^T W R W H on positions (3N x 3N) by unit JVPs at the final linearization
    obs = lv[-1][0]; cw = channel_weights(obs, terms)
    C = torch.zeros(3*n, 3*n, dtype=torch.float64); Dm = torch.zeros(3*n, 3*n, dtype=torch.float64)
    for j in range(3*n):
        e = torch.zeros(n, 3, dtype=torch.float64, device=dev); e.view(-1)[j] = 1
        hv = lin.apply_H(e)
        C[:, j] = lin.apply_HT([h*w for h, (w, _) in zip(hv, cw)]).reshape(-1).cpu()
        Dm[:, j] = lin.apply_HT([h*wr for h, (_, wr) in zip(hv, cw)]).reshape(-1).cpu()
    C, Dm = (C+C.T)/2, (Dm+Dm.T)/2
    emb = lambda M: torch.zeros(6*n, 6*n, dtype=torch.float64).index_put_((torch.tensor([6*(i//3)+i % 3 for i in range(3*n)])[:, None], torch.tensor([6*(i//3)+i % 3 for i in range(3*n)])[None, :]), M)
    Ce, De = emb(C), emb(Dm)
    lam = torch.block_diag(*torch.linalg.inv(Pm.cpu()))
    B = Gd.T@(lam+Ce)@Gd+Pi; B = (B+B.T)/2
    evB = torch.linalg.eigvalsh(B)
    Binv = torch.linalg.pinv(B, hermitian=True)
    K = Gd@Binv@Gd.T                                   # K_G = K C~ on the measured part
    A = torch.eye(6*n, dtype=torch.float64)-K@Ce
    Pblk = torch.block_diag(*Pm.cpu())
    one = A@Pblk@A.T+K@De@K.T; one = (one+one.T)/2
    Pfull = A@Pfull@A.T+K@De@K.T; Pfull = (Pfull+Pfull.T)/2
    blk = lambda M: torch.stack([M[6*i:6*i+6, 6*i:6*i+6] for i in range(n)])
    ev_one, ev_full = torch.linalg.eigvalsh(one), torch.linalg.eigvalsh(Pfull)
    ev_meth = torch.linalg.eigvalsh(P.cpu())
    row = {'frame': k, 'cond_B': float(evB[-1]/evB[0]), 'min_eig_B': float(evB[0]),
           'method_max_var': float(P.cpu().diagonal(dim1=1, dim2=2).max()), 'method_min_block_eig': float(ev_meth[:, 0].min()),
           'one_step_dense_max_var': float(blk(one).diagonal(dim1=1, dim2=2).max()), 'one_step_min_eig': float(ev_one[0]),
           'method_vs_one_step_rel': float((P.cpu()-blk(one)).norm()/blk(one).norm()),
           'full_max_var': float(Pfull.diagonal().max()), 'full_median_pos_var': float(blk(Pfull)[:, :3, :3].diagonal(dim1=1, dim2=2).median()),
           'full_min_eig': float(ev_full[0]),
           'method_vs_full_blocks_rel': float((P.cpu()-blk(Pfull)).norm()/blk(Pfull).norm()),
           'structured_max_abs_C': float(cov.C.abs().max()) if hasattr(cov, 'C') else None,
           'structured_max_abs_X': float(cov.X.abs().max()) if hasattr(cov, 'X') else None, 'psd_repair': rep['posterior']['psd_repair']['blocks_floored'], 'seconds': time.time()-clock}
    rows.append(row); print(json.dumps(row), flush=True)
a.out.parent.mkdir(parents=True, exist_ok=True); a.out.write_text(json.dumps({'points': n, 'groups': S, 'rows': rows}, indent=1))
