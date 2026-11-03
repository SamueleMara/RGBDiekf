#!/usr/bin/env python3
"""GT-ONLY physical falsification of temporal RGB-D residuals at the TRUE synthetic motion (never read by inference).

For the full frame-0 endpoint catalogue (every valid foreground pixel of every camera, original point IDs) the true
positions x_k = T_GT(q_k - q_0) x_0 are formed from the labelled part clouds and the joint metadata. For each camera
and consecutive pair (tau -> t) the following are evaluated AT THE TRUE STATES:

  (A) old fixed-current-pixel Eulerian warp (build_temporal_rows):  p = GT first-hit pixel at t,
        q = p - pi(x_t) + pi(x_tau),  r = y_t(p) - [I_tau(q), D_tau(q) + dZ],  R = R_t(p) + sum w^2 R_tau
  (B) material-following residual:  p_ref = GT first-hit pixel at tau,  q_t = p_ref + pi(x_t) - pi(x_tau),
        r = [I_t(q_t) - I_tau(p_ref), D_t(q_t) - D_tau(p_ref) - dZ],  R = sum w^2 R_t + R_tau(p_ref)
  (C) direct geometry  D_t(pi(x_t)) - Z_t(x_t)        (bilinear, metric depth)
  (D) subpixel-anchor photometric  I_t(pi(x_t)) - I_tau(pi(x_tau))
  (Z) zero motion  y_t(p_ref) - y_tau(p_ref)          (the same rows of (B) evaluated at x_t = x_tau)

Normalisation uses ONLY the sensor covariance the temporal estimator declares (scene/noise_calibration.py:
temporal_sensor_noise, the same call as train.py). No variance is guessed here. The bilinear resampling remainder is
measured separately from the data (second differences of the current image), never added to any covariance.

    python -B tools/audit_temporal_dynamic_correspondence.py --scenes USB_100109,blade_103706 --frames 0,1,2 \
        --output-root "/media/.../RGBDekf/material_warp_gt_audit"
"""
import argparse
import datetime
import json
import math
import sys
import uuid
from pathlib import Path
import numpy as np
import torch
from scipy.spatial import cKDTree
from scipy.stats import chi2
from scipy.ndimage import distance_transform_edt
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.evaluate_projective_gt import gt_joint, labelled_endpoints, gt_transform
from utils.io import json_write, serial

DEFAULT_DATA = Path('/media/samuelemara/Extreme SSD/Samuele/Backup_09_10_26/Sapien/output/preparation_8cams/partnet_subset_fixed_render_linear')
REQUESTED_TRACES = {'USB_100109': [67827, 87290, 85696], 'blade_103706': [27016]}
CHI2_4_MEDIAN, CHI2_4_99 = float(chi2.median(4)), float(chi2.isf(.01, 4))
REPO = Path(__file__).resolve().parents[1]


def q_stats(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if not len(x):
        return None
    return {'count': int(len(x)), 'median': float(np.median(x)), 'p90': float(np.quantile(x, .9)), 'p99': float(np.quantile(x, .99)),
            'mean': float(x.mean())}


def ground_truth(scene, data, xyz):
    """True positions of the catalogue points at the selected frames, plus the evidence that grounds them."""
    kind, axis, origin, declared = gt_joint(scene)
    values = np.genfromtxt(scene/'metadata/frame_values.csv', delimiter=',', names=True)
    index = np.asarray(values[values.dtype.names[0]], int)
    allq = np.asarray(values[values.dtype.names[1]], float)
    ids = list(data.frame_ids)
    if not np.array_equal(index[ids], np.asarray(ids)):
        raise ValueError('frame_values.csv index does not match dataset frame ids')
    q = allq[ids]
    (x0, l0), _, moving_label, note = labelled_endpoints(scene, kind, axis, origin, allq[-1]-allq[0])
    distance, nn = cKDTree(x0).query(xyz)
    moving = l0[nn] == moving_label
    X = np.repeat(xyz[None], len(ids), 0)
    for k in range(len(ids)):
        T = gt_transform(kind, axis, origin, q[k]-q[0])
        X[k, moving] = xyz[moving]@T[:3, :3].T+T[:3, 3]
    timing = json.loads((scene/'metadata/timing.json').read_text()) if (scene/'metadata/timing.json').exists() else {}
    evidence = {'joint_kind': kind, 'declared_type': declared, 'axis': axis.tolist(), 'origin': origin.tolist(), 'q_selected': q.tolist(),
                'dq_per_frame': np.diff(q).tolist(), 'moving_label': int(moving_label), 'label_note': note,
                'dataset_times_s': np.asarray(data.times).tolist(), 'timing_json_frame_interval_s': timing.get('frame_interval_s'),
                'label_distance_m_moving': q_stats(distance[moving]), 'label_distance_m_static': q_stats(distance[~moving]),
                'label_distance_within_1mm_fraction': float((distance < 1e-3).mean()),
                'moving_points': int(moving.sum()), 'static_points': int((~moving).sum())}
    return X, moving, distance, evidence


def verified_labels(scene, rig, noise, policy, xyz, moving, device, check_frames):
    """GT-only label verification. The sparse labelled part clouds (hundreds of vertices) cannot resolve contact
    regions, so each point's moving/static label is re-tested at distant frames: under each hypothesis the GT joint
    transform gives a position; a camera votes for a hypothesis when that position lies on the measured first surface
    (|D_k(pi(x)) - Z| within the documented single-surface allowance, valid 2x2 support). The label with strictly more
    votes wins the vote; a point's label is VERIFIED only when the vote agrees with the nearest-labelled-cloud label;
    otherwise LABEL_UNVERIFIED (excluded from class statistics, reported)."""
    from scene.dataset import Dataset
    from estimator.projective_sensor import MeasuredFrame, sample
    kind, axis, origin, _ = gt_joint(scene)
    values = np.genfromtxt(scene/'metadata/frame_values.csv', delimiter=',', names=True)
    allq = np.asarray(values[values.dtype.names[1]], float)
    data = Dataset(scene)
    votes = np.zeros((len(xyz), 2), int)
    x = torch.as_tensor(xyz, device=device)
    for k in check_frames:
        F = MeasuredFrame(data, k, device, torch.float64)
        T = torch.as_tensor(gt_transform(kind, axis, origin, allq[data.frame_ids[k]]-allq[data.frame_ids[0]]), device=device)
        hyp = (x, x@T[:3, :3].T+T[:3, 3])
        for c in range(rig.count):
            cc = torch.full((len(x),), c, device=device)
            f = float(min(rig.K[c, 0], rig.K[c, 1]))
            for h, xh in enumerate(hyp):
                u, z, _, _ = rig.project_rows(xh, cc)
                smp = sample(F, cc, u, noise, policy, rig.K)
                allow = policy['same_surface_n_sigma']*math.sqrt(2)*noise.depth_sigma2(z).sqrt()+z*math.tan(math.radians(policy['same_surface_max_incidence_deg']))*math.sqrt(2)/f
                on = (smp.reason == 0) & ((smp.depth-z).abs() <= allow)
                votes[:, h] += on.cpu().numpy()
    vote = np.where(votes[:, 1] > votes[:, 0], 1, np.where(votes[:, 0] > votes[:, 1], 0, -1))
    agree = vote == moving.astype(int)
    # identity is grounded only when the two independent label sources agree (in-plane sliding makes votes ambiguous)
    verified = np.where(agree, vote, -1)
    return verified, votes, {'check_frames': list(check_frames), 'labelled_cloud_vertices': None,
                             'nn_label_moving': int(moving.sum()), 'verified_moving': int((verified == 1).sum()),
                             'verified_static': int((verified == 0).sum()), 'unverified': int((verified < 0).sum()),
                             'nn_moving_vote_static': int((moving & (vote == 0)).sum()),
                             'nn_static_vote_moving': int((~moving & (vote == 1)).sum()), 'vote_tie_or_no_vote': int((vote < 0).sum()),
                             'agreement_fraction_among_voted': float(agree[vote >= 0].mean()),
                             'rule': 'verified = NN part-cloud label AND multi-frame on-surface vote agree; otherwise LABEL_UNVERIFIED (excluded)'}


def edge_distance(frame, c, noise, policy, K):
    """Pixel distance to the nearest foreground boundary or same-surface depth break (frame t, camera c)."""
    m = (frame.mask[c] & frame.raw_valid[c]).cpu().numpy()
    D = frame.depth[c].double().cpu().numpy()
    edge = np.zeros_like(m)
    f = float(min(K[c, 0], K[c, 1]))
    for dy, dx in ((0, 1), (1, 0)):
        a, b = m[:m.shape[0]-dy, :m.shape[1]-dx], m[dy:, dx:]
        Da, Db = D[:D.shape[0]-dy, :D.shape[1]-dx], D[dy:, dx:]
        z = np.where(a & b, (Da+Db)/2, 1.)
        sd = np.sqrt(noise.depth_variance+noise.depth_squared_variance*z*z)
        allow = policy['same_surface_n_sigma']*math.sqrt(2)*sd+z*math.tan(math.radians(policy['same_surface_max_incidence_deg']))*math.sqrt(2)/f
        brk = (a != b) | (a & b & (np.abs(Da-Db) > allow))
        edge[:m.shape[0]-dy, :m.shape[1]-dx] |= brk
        edge[dy:, dx:] |= brk
    return distance_transform_edt(~edge)


def second_difference_remainder(frame, c, q):
    """Data-only estimate of (bilinear sample - underlying smooth image) at q: +1/2 [du(1-du) F_uu + dv(1-dv) F_vv] from
    3-point second differences around the 2x2 cell (Taylor remainder of linear interpolation; diagnostic, not a covariance)."""
    H, W = frame.H, frame.W
    base = torch.floor(q)
    du, dv = (q-base).unbind(-1)
    x0, y0 = base[:, 0].long(), base[:, 1].long()
    ok = (x0 >= 1) & (y0 >= 1) & (x0+2 <= W-1) & (y0+2 <= H-1)
    x0, y0 = x0.clamp(1, W-3), y0.clamp(1, H-3)
    cc = torch.full_like(x0, c)
    F = torch.cat((frame.rgb[c].double(), frame.depth[c].double()[..., None]), -1)
    g = lambda dy, dx: F[y0+dy, x0+dx]
    Fuu = ((g(0, -1)-2*g(0, 0)+g(0, 1))+(g(0, 0)-2*g(0, 1)+g(0, 2)))/2
    Fvv = ((g(-1, 0)-2*g(0, 0)+g(1, 0))+(g(0, 0)-2*g(1, 0)+g(2, 0)))/2
    rem = .5*((du*(1-du))[:, None]*Fuu+(dv*(1-dv))[:, None]*Fvv)
    mk = frame.mask[cc, y0, x0]
    for dy, dx in ((0, -1), (0, 2), (-1, 0), (2, 0), (1, 1), (0, 1), (1, 0)):
        mk = mk & frame.mask[cc, y0+dy, x0+dx]
    return rem, ok & mk


def audit_pair(scene_name, data, rig, noise, policy, frames, X, moving, a, b, device, keep_trace, source_cam, excluded):
    from estimator.projective_sensor import sample
    from estimator.temporal_zbuffer import first_hit_pixel_owner
    Fa, Fb = frames[a], frames[b]
    xa = torch.as_tensor(X[a], device=device)
    xb = torch.as_tensor(X[b], device=device)
    own_a = first_hit_pixel_owner(xa, rig, Fa, dense=False)
    own_b = first_hit_pixel_owner(xb, rig, Fb, dense=False)
    mv = torch.as_tensor(moving, device=device)
    N, C, W = len(X[0]), rig.count, Fa.W
    cams, per_point = [], {name: torch.zeros(N, dtype=torch.float64, device=device) for name in ('J_true', 'J_zero', 'rows')}
    info = torch.zeros((N, 3, 3), dtype=torch.float64, device=device)
    info_rgb = torch.zeros_like(info)
    rows_out = []
    for c in range(C):
        cc = lambda n: torch.full((n,), c, dtype=torch.long, device=device)
        ua, za, Ja, Jza = rig.project_rows(xa, cc(N))
        ub, zb, Jb, Jzb = rig.project_rows(xb, cc(N))
        pa = own_a.pixel[c]
        ya, xa_ = pa//W, pa % W
        eligible = (own_a.reason[c] == 0) & Fa.mask[c].reshape(-1)[pa.clamp(0, Fa.H*W-1)] & Fa.raw_valid[c].reshape(-1)[pa.clamp(0, Fa.H*W-1)]
        i = torch.where(eligible)[0]
        n = len(i)
        pref = torch.stack((xa_[i], ya[i]), -1).double()
        du = ub[i]-ua[i]
        dz = zb[i]-za[i]
        yref = torch.cat((Fa.rgb[c, ya[i], xa_[i]].double(), Fa.depth[c, ya[i], xa_[i]].double()[:, None]), -1)
        Rref = torch.cat((noise.rgb_variance.expand(n, 3), noise.depth_sigma2(yref[:, 3])[:, None]), -1)
        # (B) material-following at the true motion
        qt = pref+du
        sb = sample(Fb, cc(n), qt, noise, policy, rig.K)
        hB = torch.cat((sb.rgb, sb.depth[:, None]), -1)
        rB = hB-yref
        rB[:, 3] -= dz
        RB = torch.cat((sb.var_rgb, sb.var_depth[:, None]), -1)+Rref
        nisB = (rB*rB/RB).sum(1)
        # (Z) zero motion on the same material rows
        y0b = torch.cat((Fb.rgb[c, ya[i], xa_[i]].double(), Fb.depth[c, ya[i], xa_[i]].double()[:, None]), -1)
        okZ = Fb.mask[c, ya[i], xa_[i]] & Fb.raw_valid[c, ya[i], xa_[i]]
        rZ = y0b-yref
        RZ = torch.cat((noise.rgb_variance.expand(n, 3), noise.depth_sigma2(y0b[:, 3])[:, None]), -1)+Rref
        nisZ = (rZ*rZ/RZ).sum(1)
        # (C), (D) at the true subpixel projections
        sbt = sample(Fb, cc(n), ub[i], noise, policy, rig.K)
        sat = sample(Fa, cc(n), ua[i], noise, policy, rig.K)
        rC = sbt.depth-zb[i]
        rCa = sat.depth-za[i]
        rD = sbt.rgb-sat.rgb
        okCD = (sbt.reason == 0) & (sat.reason == 0)
        # visibility of the true state at t
        winner_b = own_b.reason[c, i] == 0
        meas_front = torch.zeros(n, dtype=torch.bool, device=device)
        pb = own_b.pixel[c, i].clamp(0, Fb.H*W-1)
        inside_b = own_b.reason[c, i] != 2
        Dpb = Fb.depth[c].reshape(-1)[pb].double()
        mpb = Fb.mask[c].reshape(-1)[pb] & Fb.raw_valid[c].reshape(-1)[pb]
        f = float(min(rig.K[c, 0], rig.K[c, 1]))
        allow = policy['same_surface_n_sigma']*math.sqrt(2)*noise.depth_sigma2(zb[i]).sqrt()+zb[i]*math.tan(math.radians(policy['same_surface_max_incidence_deg']))*math.sqrt(2)/f
        meas_front = inside_b & mpb & (zb[i]-Dpb > allow)
        validB = sb.reason == 0
        # reference identity: the measured source pixel must show the point's own surface at tau (same documented
        # single-surface allowance); the true state must lie on the measured first surface at t (GT-only visibility)
        allow_a = policy['same_surface_n_sigma']*math.sqrt(2)*noise.depth_sigma2(za[i]).sqrt()+za[i]*math.tan(math.radians(policy['same_surface_max_incidence_deg']))*math.sqrt(2)/f
        ref_gap = yref[:, 3]-za[i]
        ref_same = ref_gap.abs() <= allow_a
        cur_on_surface = (sbt.reason == 0) & (rC.abs() <= allow)
        visible = validB & ~meas_front & ref_same & cur_on_surface
        nonsource = torch.as_tensor(source_cam[i.cpu().numpy()] != c, device=device)
        # data-only resampling remainder and context
        rem, okrem = second_difference_remainder(Fb, c, qt)
        ed = torch.as_tensor(edge_distance(Fb, c, noise, policy, rig.K.cpu().numpy()), device=device)
        qi = torch.round(torch.nan_to_num(qt)).long()
        qi[:, 0].clamp_(0, W-1)
        qi[:, 1].clamp_(0, Fb.H-1)
        dist_edge = ed[qi[:, 1], qi[:, 0]]
        tan_incidence = sb.grad_depth.norm(dim=1)*f/sb.depth.clamp_min(1e-9)
        grad_snr = sb.grad_rgb.norm(dim=2).max(1).values/noise.rgb_variance.max().sqrt()
        # (A) old fixed-current-pixel warp for GT owners at t that have this camera's GT reference at tau
        ownb = (own_b.reason[c] == 0)
        pbb = own_b.pixel[c].clamp(0, Fb.H*W-1)
        ownb &= Fb.mask[c].reshape(-1)[pbb] & Fb.raw_valid[c].reshape(-1)[pbb] & eligible
        j = torch.where(ownb)[0]
        pj = own_b.pixel[c, j]
        uvp = torch.stack((pj % W, pj//W), -1).double()
        qA = uvp-ub[j]+ua[j]
        sa = sample(Fa, cc(len(j)), qA, noise, policy, rig.K)
        yb = torch.cat((Fb.rgb[c, pj//W, pj % W].double(), Fb.depth[c, pj//W, pj % W].double()[:, None]), -1)
        hA = torch.cat((sa.rgb, (sa.depth+zb[j]-za[j])[:, None]), -1)
        rA = yb-hA
        RA = torch.cat((noise.rgb_variance.expand(len(j), 3), noise.depth_sigma2(yb[:, 3])[:, None]), -1)+torch.cat((sa.var_rgb, sa.var_depth[:, None]), -1)
        nisA = (rA*rA/RA).sum(1)
        validA = sa.reason == 0
        # observability of x_t through (B) (true linearization)
        JB = torch.zeros((n, 4, 3), dtype=torch.float64, device=device)
        JB[:, :3] = sb.grad_rgb@Jb[i]
        JB[:, 3] = (sb.grad_depth[:, None, :]@Jb[i])[:, 0]-Jzb[i]
        w = torch.where(visible[:, None], 1/RB, torch.zeros_like(RB))
        info.index_add_(0, i, torch.einsum('mri,mr,mrj->mij', JB, w, JB))
        info_rgb.index_add_(0, i, torch.einsum('mri,mr,mrj->mij', JB[:, :3], w[:, :3], JB[:, :3]))
        both = visible & okZ
        per_point['J_true'].index_add_(0, i[both], nisB[both])
        per_point['J_zero'].index_add_(0, i[both], nisZ[both])
        per_point['rows'].index_add_(0, i[both], torch.ones(int(both.sum()), dtype=torch.float64, device=device))
        mvi, mvj = mv[i], mv[j]
        rec = {'camera': c, 'camera_name': rig.names[c]}
        exi, exj = excluded[i], excluded[j]
        for name, g, gA in (('moving', mvi & ~exi, mvj & ~exj), ('static', ~mvi & ~exi, ~mvj & ~exj)):
            vis = g & visible
            core = vis & (dist_edge > 2) & (grad_snr >= 1)
            r = {'eligible_reference_rows': int(g.sum()),
                 'true_state_not_first_hit_at_t': int((g & ~winner_b).sum()),
                 'measured_surface_in_front_occluded': int((g & meas_front).sum()),
                 'reference_pixel_shows_other_surface': int((g & ~ref_same).sum()),
                 'reference_gap_abs_m': q_stats(ref_gap[g].abs().cpu().numpy()),
                 'true_state_off_measured_surface_at_t': int((g & (sbt.reason == 0) & ~(rC.abs() <= allow)).sum()),
                 'four_neighbor_invalid': {name2: int((g & (sb.reason == code)).sum()) for name2, code in
                                           (('outside_image', 2), ('mask', 3), ('invalid_depth', 4), ('depth_discontinuity', 5))},
                 'visible_valid_rows': int(vis.sum()),
                 'visible_but_not_GT_first_hit': int((vis & ~winner_b).sum()),
                 'textureless_rgb_gradient_below_one_sigma': int((vis & (grad_snr < 1)).sum()),
                 'within_2px_of_edge': int((vis & (dist_edge <= 2)).sum()),
                 'source_current_pixel_shift_px': q_stats(du[g].norm(dim=1).cpu().numpy()),
                 'B_material': residual_block(rB[vis], nisB[vis], RB[vis]),
                 'B_material_interior_textured': residual_block(rB[core], nisB[core], RB[core]),
                 'B_minus_data_remainder_interior': residual_block((rB-rem)[core & okrem], None, None),
                 'data_remainder_interior_abs': q_stats(rem[core & okrem, 3].abs().cpu().numpy()),
                 'Z_zero_motion_same_rows': residual_block(rZ[vis & okZ], nisZ[vis & okZ], RZ[vis & okZ]),
                 'A_old_fixed_pixel': residual_block(rA[gA & validA], nisA[gA & validA], RA[gA & validA]),
                 'A_rows': int((gA & validA).sum()), 'A_reference_support_invalid': int((gA & ~validA).sum()),
                 'C_depth_geometry_t_m': q_stats(rC[g & okCD].abs().cpu().numpy()),
                 'C_depth_geometry_t_signed_median_m': float(rC[g & okCD].median()) if bool((g & okCD).any()) else None,
                 'C_depth_geometry_tau_m': q_stats(rCa[g & okCD].abs().cpu().numpy()),
                 'D_subpixel_rgb_abs': q_stats(rD[g & okCD].abs().max(1).values.cpu().numpy()),
                 'C_rows': int((g & okCD).sum()),
                 'C_nonsource_t_abs_m': q_stats(rC[g & okCD & nonsource].abs().cpu().numpy()),
                 'C_nonsource_tau_abs_m': q_stats(rCa[g & okCD & nonsource].abs().cpu().numpy())}
            rec[name] = r
        cams.append(rec)
        sel = torch.where(keep_trace[i])[0]
        for t in sel.tolist():
            rows_out.append({'point': int(i[t]), 'camera': c, 'p_ref_uv': pref[t].tolist(), 'u_tau': ua[i[t]].tolist(), 'u_t': ub[i[t]].tolist(),
                             'q_t': qt[t].tolist(), 'delta_u_px': du[t].tolist(), 'delta_Z_m': float(dz[t]), 'reason_B': int(sb.reason[t]),
                             'visible': bool(visible[t]), 'GT_first_hit_at_t': bool(winner_b[t]), 'measured_surface_in_front': bool(meas_front[t]),
                             'reference_same_surface': bool(ref_same[t]), 'reference_gap_m': float(ref_gap[t]), 'reference_depth_m': float(yref[t, 3]),
                             'r_B': rB[t].tolist(), 'NIS_B': float(nisB[t]), 'r_zero_motion': rZ[t].tolist(), 'NIS_zero': float(nisZ[t]),
                             'r_C_depth_m': float(rC[t]), 'r_D_rgb': rD[t].tolist(), 'distance_to_edge_px': float(dist_edge[t]),
                             'rgb_gradient_snr': float(grad_snr[t]), 'tan_incidence_proxy': float(tan_incidence[t]),
                             'data_remainder': rem[t].tolist(), 'R_B_diag': RB[t].tolist()})
        if not hasattr(audit_pair, 'scatter'):
            audit_pair.scatter = {}
        vis_m = visible & mvi & ~exi
        audit_pair.scatter.setdefault((scene_name, a, b), []).append(
            {'edge': dist_edge[vis_m].cpu().numpy(), 'tan': tan_incidence[vis_m].cpu().numpy(), 'rD': rB[vis_m, 3].cpu().numpy(),
             'rI': rB[vis_m, :3].abs().max(1).values.cpu().numpy(), 'rem': rem[vis_m, 3].cpu().numpy(), 'okrem': okrem[vis_m].cpu().numpy()})
    # per-point likelihood comparison truth vs zero motion and true-motion information
    disp = xb-xa
    ev = torch.as_tensor(np.linalg.eigvalsh(info.cpu().numpy()), device=device)
    seen = per_point['rows'] > 0
    zscore2 = torch.einsum('ni,nij,nj->n', disp, info, disp)
    zscore2_rgb = torch.einsum('ni,nij,nj->n', disp, info_rgb, disp)
    points = {}
    for name, g in (('moving', mv & ~excluded), ('static', ~mv & ~excluded)):
        h = g & seen
        points[name] = {'points_with_visible_rows': int(h.sum()), 'points_without_visible_rows': int((g & ~seen).sum()),
                        'truth_preferred_over_zero_motion_fraction': float((per_point['J_true'][h] < per_point['J_zero'][h]).double().mean()) if bool(h.any()) else None,
                        'J_true_per_row': q_stats((per_point['J_true'][h]/per_point['rows'][h]).cpu().numpy()),
                        'J_zero_per_row': q_stats((per_point['J_zero'][h]/per_point['rows'][h]).cpu().numpy()),
                        'information_eigenvalues_per_m2_min_mid_max': [q_stats(ev[h, j].cpu().numpy()) for j in range(3)],
                        'weakest_direction_std_mm': q_stats(1e3/ev[h, 0].clamp_min(1e-300).sqrt().cpu().numpy()),
                        'true_displacement_mm': q_stats(1e3*disp[h].norm(dim=1).cpu().numpy()),
                        'true_displacement_information_zscore': q_stats(zscore2[h].clamp_min(0).sqrt().cpu().numpy()),
                        'true_displacement_rgb_only_zscore': q_stats(zscore2_rgb[h].clamp_min(0).sqrt().cpu().numpy())}
    return {'from_frame': a, 'to_frame': b, 'cameras': cams, 'points': points,
            'owner_conflicts_tau_t': [own_a.conflict_count, own_b.conflict_count]}, rows_out


def residual_block(r, nis, R):
    if r is None or not len(r):
        return None
    out = {'rows': int(len(r)), 'abs_rgb_max_channel': q_stats(r[:, :3].abs().max(1).values.cpu().numpy()),
           'abs_depth_m': q_stats(r[:, 3].abs().cpu().numpy()),
           'signed_depth_median_m': float(r[:, 3].median()), 'signed_depth_mean_m': float(r[:, 3].mean()),
           'signed_rgb_mean': r[:, :3].mean(0).tolist()}
    if nis is not None:
        z = r/R.sqrt()
        out.update({'NIS': q_stats(nis.cpu().numpy()), 'NIS_median_over_chi2_4_median': float(nis.median())/CHI2_4_MEDIAN,
                    'large_NIS_above_chi2_4_99_fraction': float((nis > CHI2_4_99).double().mean()),
                    'standardized_abs_median_rgb_depth': z.abs().median(0).values.tolist(),
                    'standardized_signed_mean_rgb_depth': z.mean(0).tolist()})
    return out


def figures(out, scene_name, data, rig, frames, X, moving, traces, pair_scatter):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig_dir = out/'figures'
    fig_dir.mkdir(exist_ok=True)
    names = []
    for (sc, a, b), chunks in pair_scatter.items():
        cat = {k: np.concatenate([c[k] for c in chunks]) for k in chunks[0]}
        fig, ax = plt.subplots(1, 3, figsize=(15, 4))
        for axis, key, label in ((ax[0], 'edge', 'distance to mask/depth edge (px)'), (ax[1], 'tan', 'tan(incidence) proxy |grad D| f / Z')):
            x, y = cat[key], np.abs(cat['rD'])
            ok = np.isfinite(x) & np.isfinite(y) & (y > 0)
            axis.scatter(x[ok], y[ok], s=1, alpha=.25)
            axis.set_yscale('log')
            axis.set_xlabel(label)
            axis.set_ylabel('|r_D| material-following at GT (m)')
        ok = cat['okrem'] & np.isfinite(cat['rem'])
        ax[2].scatter(np.abs(cat['rem'][ok])+1e-12, np.abs(cat['rD'][ok])+1e-12, s=1, alpha=.25)
        ax[2].set_xscale('log')
        ax[2].set_yscale('log')
        ax[2].plot([1e-9, 1e-2], [1e-9, 1e-2], 'k--', lw=.8)
        ax[2].set_xlabel('data-only bilinear remainder |.| (m)')
        ax[2].set_ylabel('|r_D| (m)')
        fig.suptitle(f'{sc} moving visible rows, frames {a}->{b} (GT-only diagnostic)')
        fig.tight_layout()
        name = f'{sc}_{a}_{b}_residual_vs_context.png'
        fig.savefig(fig_dir/name, dpi=110)
        plt.close(fig)
        names.append(name)
    for pid, recs in traces.items():
        recs = [r for r in recs if r['visible']][:2] or recs[:2]
        if not recs:
            continue
        fig, ax = plt.subplots(len(recs), 4, figsize=(16, 4*len(recs)), squeeze=False)
        for row, r in enumerate(recs):
            c, a, b = r['camera'], r['from_frame'], r['to_frame']
            cx, cy = int(round(r['p_ref_uv'][0])), int(round(r['p_ref_uv'][1]))
            x0, y0 = max(cx-7, 0), max(cy-7, 0)
            for col, (k, kind) in enumerate(((a, 'rgb'), (b, 'rgb'), (a, 'depth'), (b, 'depth'))):
                F = frames[k]
                img = F.rgb[c].cpu().numpy() if kind == 'rgb' else np.where(F.mask[c].cpu().numpy(), F.depth[c].cpu().numpy(), np.nan)
                crop = img[y0:y0+15, x0:x0+15]
                ax[row, col].imshow(np.clip(crop, 0, 1) if kind == 'rgb' else crop, interpolation='nearest', extent=(x0-.5, x0+crop.shape[1]-.5, y0+crop.shape[0]-.5, y0-.5))
                ax[row, col].plot(*r['p_ref_uv'], 'g+', ms=12, label='p_ref (measured source pixel)')
                ax[row, col].plot(*r['u_tau'], 'rx', label='GT pi(x_tau)')
                ax[row, col].plot(*r['u_t'], 'mx', label='GT pi(x_t)')
                ax[row, col].plot(*r['q_t'], 'co', fillstyle='none', ms=10, label='q_t = p_ref + du')
                ax[row, col].set_title(f'cam {c} frame {k} {kind}')
            ax[row, 0].legend(fontsize=6, loc='lower left')
            ax[row, 3].text(0.02, .02, f"NIS_B={r['NIS_B']:.3g} NIS_0={r['NIS_zero']:.3g}\nr_B={np.round(r['r_B'], 6).tolist()}\nedge={r['distance_to_edge_px']:.1f}px",
                            transform=ax[row, 3].transAxes, fontsize=7, color='w', bbox={'facecolor': 'k', 'alpha': .6})
        fig.suptitle(f'{scene_name} point {pid} ({"MOVING" if moving[pid] else "static"}) GT-only audit')
        fig.tight_layout()
        name = f'{scene_name}_point_{pid}.png'
        fig.savefig(fig_dir/name, dpi=90)
        plt.close(fig)
        names.append(name)
    return names


def correspondence_verdict(evidence, pairs):
    """Grounded when (i) source pixels reproject exactly at frame 0 and (ii) on non-source cameras the true moving
    states land on the measured surface at t as closely as the SAME rows did at tau (both are subpixel bilinear samples
    of one rendered surface), median ratio <= 2 per pair."""
    if evidence['source_reprojection_px']['p99'] > 1e-6 or evidence['source_depth_reprojection_m']['p99'] > 1e-9:
        return {'status': 'GT_CORRESPONDENCE_UNVERIFIED', 'reason': 'source reprojection'}
    ratios = []
    for p in pairs:
        t = [c['moving']['C_nonsource_t_abs_m'] for c in p['cameras'] if c['moving']['C_nonsource_t_abs_m']]
        a = [c['moving']['C_nonsource_tau_abs_m'] for c in p['cameras'] if c['moving']['C_nonsource_tau_abs_m']]
        if not t:
            return {'status': 'GT_CORRESPONDENCE_UNVERIFIED', 'reason': 'no non-source moving rows'}
        mt = float(np.average([x['median'] for x in t], weights=[x['count'] for x in t]))
        ma = float(np.average([x['median'] for x in a], weights=[x['count'] for x in a]))
        ratios.append({'pair': [p['from_frame'], p['to_frame']], 'moving_abs_C_t_median_m': mt, 'moving_abs_C_tau_median_m': ma, 'ratio': mt/max(ma, 1e-300)})
    ok = all(r['ratio'] <= 2 for r in ratios)
    return {'status': 'VERIFIED' if ok else 'GT_CORRESPONDENCE_UNVERIFIED', 'evidence': ratios,
            'rule': 'exact source reprojection and non-source |D_t(pi(x_t))-Z_t| median <= 2x the same rows at tau'}


def gate(scene_result):
    """Gate A on genuinely visible, same-surface, interior, textured moving rows of (B)."""
    verdicts = []
    for pair in scene_result['pairs']:
        nis, dep, rgb, rows = [], [], [], 0
        for cam in pair['cameras']:
            blk = cam['moving']['B_material_interior_textured']
            if blk:
                rows += blk['rows']
                nis.append((blk['rows'], blk['NIS']['median']))
                dep.append((blk['rows'], blk['signed_depth_median_m'], blk['abs_depth_m']['median']))
        if not rows:
            verdicts.append({'pair': [pair['from_frame'], pair['to_frame']], 'status': 'NO_VISIBLE_INTERIOR_MOVING_ROWS'})
            continue
        med = float(np.average([m for _, m in nis], weights=[n for n, _ in nis]))
        verdicts.append({'pair': [pair['from_frame'], pair['to_frame']], 'rows': rows, 'row_weighted_camera_NIS_median': med,
                         'over_chi2_4_median': med/CHI2_4_MEDIAN,
                         'row_weighted_abs_depth_median_m': float(np.average([d for _, _, d in dep], weights=[n for n, _, _ in dep])),
                         'row_weighted_signed_depth_median_m': float(np.average([s for _, s, _ in dep], weights=[n for n, _, _ in dep]))})
    return verdicts


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--scenes', default='USB_100109,blade_103706')
    ap.add_argument('--frames', default='0,1,2')
    ap.add_argument('--data-root', type=Path, default=DEFAULT_DATA)
    ap.add_argument('--output-root', type=Path, required=True, help='parent; a unique audit_<stamp>_<id> child is created (never overwritten)')
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--traces', type=int, default=24, help='number of seeded-random visible moving points traced per scene (plus requested IDs and static references)')
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()
    from scene.dataset import Dataset
    from scene.config import load_config
    from scene.endpoint import endpoint_samples
    from estimator.projective_camera import CameraRig
    from estimator.projective_sensor import MeasuredFrame, SensorNoise
    from estimator.temporal_noise import temporal_sensor_only
    frame_list = [int(x) for x in a.frames.split(',')]
    if frame_list != list(range(frame_list[0], frame_list[-1]+1)) or frame_list[0] != 0:
        ap.error('--frames must be consecutive from 0 (the forward endpoint catalogue frame)')
    out = a.output_root/f'audit_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}_{uuid.uuid4().hex[:8]}'
    out.mkdir(parents=True)
    policy = json.loads((REPO/'configs/projective_rgbd.json').read_text())
    dev = torch.device(a.device)
    report = {'ground_truth_namespace': 'POSTHOC_GT_AUDIT_ONLY', 'read_by_inference': False, 'chi2_4_median': CHI2_4_MEDIAN,
              'chi2_4_99': CHI2_4_99, 'policy_file': 'configs/projective_rgbd.json', 'scenes': {}}
    for name in [s for s in a.scenes.split(',') if s]:
        scene = a.data_root/name
        data = Dataset(scene, len(frame_list))
        cfg, pol, calib = temporal_sensor_only(data, load_config(), policy)
        noise = SensorNoise.from_config(cfg, pol, dev, torch.float64)
        rig = CameraRig(data.cameras, dev, torch.float64)
        k0, _, xyz, rgb, cov, source = endpoint_samples(data, 'first', cfg)
        X, nn_moving, label_distance, evidence = ground_truth(scene, data, xyz)
        total = len(Dataset(scene).times)
        verified, votes, label_report = verified_labels(scene, rig, noise, pol, xyz, nn_moving, dev, sorted({total//2, total-1}))
        with open(scene/'pointcloud/points3D_parts.ply', 'rb') as fh:
            label_report['labelled_cloud_vertices'] = next(int(l.split()[-1]) for l in fh if l.startswith(b'element vertex'))
        evidence['label_verification'] = label_report
        moving = verified == 1
        # positions follow the verified label; unverified points keep the NN label for positions but are excluded below
        kind, axis, origin, _ = gt_joint(scene)
        q = np.asarray(evidence['q_selected'])
        for k in range(len(q)):
            T = gt_transform(kind, axis, origin, q[k]-q[0])
            X[k] = np.where((verified == 1)[:, None], xyz@T[:3, :3].T+T[:3, 3], np.where((verified == 0)[:, None], xyz, X[k]))
        excluded = verified < 0
        frames = {k: MeasuredFrame(data, k, dev, torch.float64) for k in frame_list}
        # correspondence grounding: source pixels re-project onto themselves at frame 0 with their measured depth
        sc = torch.as_tensor(source[:, 1], device=dev)
        u0, z0, _, _ = rig.project_rows(torch.as_tensor(xyz, device=dev), sc)
        pix = torch.as_tensor(source[:, 2], device=dev)
        reproj = (u0-torch.stack((pix % frames[0].W, pix//frames[0].W), -1).double()).norm(dim=1)
        dsrc = (frames[0].depth[sc, pix//frames[0].W, pix % frames[0].W].double()-z0).abs()
        evidence.update({'source_reprojection_px': q_stats(reproj.cpu().numpy()), 'source_depth_reprojection_m': q_stats(dsrc.cpu().numpy()),
                         'depth_convention': 'optical-axis Z (verified by source depth reprojection and (C) at frame 0)',
                         'catalogue_points': int(len(xyz))})
        evidence['label_distance_semantics'] = ('nearest labelled-part-cloud point; the part clouds are sparse, so this is a label '
                                                'assignment distance, NOT a correspondence residual (correspondence is tested by (C))')
        rng = np.random.default_rng(a.seed)
        keep = np.zeros(len(xyz), bool)
        requested = [i for i in REQUESTED_TRACES.get(name, []) if i < len(xyz)]
        present = [i for i in requested if moving[i] and not excluded[i]]
        keep[requested] = True
        mv_ids = np.flatnonzero(moving & ~excluded)
        keep[rng.choice(mv_ids, min(a.traces, len(mv_ids)), replace=False)] = True
        st_ids = np.flatnonzero(~moving & ~excluded)
        keep[rng.choice(st_ids, min(6, len(st_ids)), replace=False)] = True
        keep_t = torch.as_tensor(keep, device=dev)
        pairs, traces = [], {int(i): [] for i in np.flatnonzero(keep)}
        audit_pair.scatter = {}
        for ka, kb in zip(frame_list[:-1], frame_list[1:]):
            res, rows = audit_pair(name, data, rig, noise, pol, frames, X, moving, ka, kb, dev, keep_t, source[:, 1], torch.as_tensor(excluded, device=dev))
            pairs.append(res)
            for r in rows:
                traces[r['point']].append({**r, 'from_frame': ka, 'to_frame': kb})
        sdir = out/name
        sdir.mkdir()
        figs = figures(sdir, name, data, rig, frames, X, moving, traces, audit_pair.scatter)
        result = {'ground_truth': evidence, 'GT_correspondence': correspondence_verdict(evidence, pairs),
                  'sensor_noise': {'values': calib['result'], 'provenance': 'scene/noise_calibration.py:temporal_sensor_noise (same call as train.py)',
                                   'qualification': calib['qualification'], 'detail': calib['sensor']},
                  'warp_discrepancy_assumed': 'NONE (R_warp = 0: falsification of the residual against sensor noise alone)',
                  'pairs': pairs, 'requested_trace_ids': requested, 'requested_trace_ids_moving_and_present': present,
                  'figures': figs}
        result['gate_A'] = gate(result)
        json_write(sdir/'traces.json', serial({str(k): {'moving': bool(moving[k]), 'label_distance_m': float(label_distance[k]),
                                                         'nn_label_moving': bool(nn_moving[k]), 'label_votes_static_moving': votes[k].tolist(),
                                                         'source': source[k].tolist(), 'records': v} for k, v in traces.items()}))
        json_write(sdir/'audit.json', serial(result))
        report['scenes'][name] = result
        print(f'[audit] {name}: correspondence={result["GT_correspondence"]["status"]} gate_A={json.dumps(result["gate_A"])}', flush=True)
    json_write(out/'audit_summary.json', serial(report))
    write_html(out, report)
    print(out)


def write_html(out, report):
    rows = []
    for name, res in report['scenes'].items():
        rows.append(f'<h2>{name}</h2><p>GT correspondence: <b>{res["GT_correspondence"]["status"]}</b> {json.dumps(res["GT_correspondence"].get("evidence"))}; sensor noise: {res["sensor_noise"]["values"]}</p>')
        rows.append('<pre>'+json.dumps(res['gate_A'], indent=1)+'</pre>')
        rows.append('<table border=1 cellpadding=3><tr><th>pair</th><th>cam</th><th>class</th><th>eligible</th><th>visible</th><th>occluded</th>'
                    '<th>ref other surface</th><th>support invalid</th><th>shift px med</th><th>(B) NIS med</th><th>(B) |rD| med m</th><th>(B) |rI| med</th>'
                    '<th>(A) NIS med</th><th>(A) |rD| med m</th><th>(Z) NIS med</th><th>(C) |D-Z| med m</th></tr>')
        for p in res['pairs']:
            for cam in p['cameras']:
                for cls in ('moving', 'static'):
                    r = cam[cls]
                    g = lambda blk, *ks: (lambda v: f'{v:.3g}' if isinstance(v, float) else '-')(_dig(r.get(blk), ks))
                    rows.append(f'<tr><td>{p["from_frame"]}-&gt;{p["to_frame"]}</td><td>{cam["camera"]}</td><td>{cls}</td>'
                                f'<td>{r["eligible_reference_rows"]}</td><td>{r["visible_valid_rows"]}</td><td>{r["measured_surface_in_front_occluded"]}</td>'
                                f'<td>{r["reference_pixel_shows_other_surface"]}</td><td>{sum(r["four_neighbor_invalid"].values())}</td><td>{g("source_current_pixel_shift_px", "median")}</td>'
                                f'<td>{g("B_material", "NIS", "median")}</td><td>{g("B_material", "abs_depth_m", "median")}</td>'
                                f'<td>{g("B_material", "abs_rgb_max_channel", "median")}</td><td>{g("A_old_fixed_pixel", "NIS", "median")}</td>'
                                f'<td>{g("A_old_fixed_pixel", "abs_depth_m", "median")}</td><td>{g("Z_zero_motion_same_rows", "NIS", "median")}</td>'
                                f'<td>{g("C_depth_geometry_t_m", "median")}</td></tr>')
        rows.append('</table>')
        for p in res['pairs']:
            rows.append(f'<h3>points {p["from_frame"]}-&gt;{p["to_frame"]}</h3><pre>'+json.dumps(p['points'], indent=1)+'</pre>')
        rows.extend(f'<div><img src="{name}/figures/{f}" style="max-width:100%"></div>' for f in res['figures'])
    html = ('<!doctype html><meta charset="utf-8"><title>GT warp audit</title><body style="font-family:sans-serif">'
            '<h1>GT-only temporal RGB-D residual audit</h1><p>Ground truth used here only; never read by inference. '
            f'chi2_4 median {CHI2_4_MEDIAN:.3f}, 99% {CHI2_4_99:.2f}.</p>'+'\n'.join(rows)+'</body>')
    (out/'index.html').write_text(html)


def _dig(d, ks):
    for k in ks:
        if not isinstance(d, dict):
            return None
        d = d.get(k)
    return d


if __name__ == '__main__':
    raise SystemExit(main())
