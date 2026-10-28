"""Measurement covariance R MEASURED from the scene's own RGB-D frames (audit 2026-10-28 item 10): no ground truth, no
motion model, no result of the filters, nothing tuned to suppress motion.

1. Sensor noise (temporal). For evenly spaced consecutive pairs (k, k+1) of every camera, pixels valid (hard mask and
   raw depth) in both frames give d = F_k+1 - F_k = n_k+1 - n_k for the static majority, var(d) = 2 sigma^2. The
   moving minority are outliers of d, so the scale is the robust MAD: sigma^2 = (1.4826 MAD(d))^2 / 2.
     RGB : one variance per channel, pooled over cameras and pairs.
     depth: MAD per depth-quantile bin, sigma_D^2(z) = a + b z^2 fitted by non-negative least squares on the bins.
   Floors: the quantization variance q^2/12 of the stored values (q = 1/255 for 8-bit RGB, the smallest positive step
   of the depth file), so a noiseless render does not produce R = 0.
2. Model mismatch (same instant, cross-camera). Each endpoint cloud (frame 0 and frame T, back-projected from one
   camera) is sampled in every OTHER camera at the SAME frame: nothing moved, so the residual r = [a - I ; Z - D] has
   variance = sensor (through the interpolation, plus the canonical pixel's own noise) + model mismatch (view-dependent
   colour, exposure / white balance between cameras, calibration, the canonical colour being a pixel-footprint average
   while the other camera is sampled bilinearly at a sub-pixel location). The additive model variance m >= 0 per block
   (RGB pooled, depth) solves median(r^2 / (v_sensor + m)) = median(chi2_1): a robust moment match, insensitive to the
   occluded / wrong-surface rows that the filter gates anyway. Rows: front-most point of their pixel, valid 2x2 sample,
   measured surface not in front (projective_terms 'depth_front_only' rule).
   The canonical point's back-projection error at that instant (H Sigma_x H^T) is NOT credited: it cannot be separated
   from the pixel-footprint appearance term, and unlike the position error that term does not shrink as the filter
   refines x (crediting it gave m = 0 on a ray-cast fixture and the row gate then rejected most rows of exactly static
   points). m is therefore an upper bound of the persistent mismatch at the endpoint geometry, stated as such.
The result replaces config.color_covariance, depth_variance, depth_squared_variance and policy rgb/depth_model_variance
for every stage; the endpoint back-projection covariance (scene/endpoint.py) uses the measured depth noise too.
"""
import copy
import numpy as np
import torch
from scipy.optimize import nnls
from scipy.stats import chi2

MAD = 1.4826


def _mad_var(d):
    d = np.asarray(d, float)
    return float((MAD*np.median(np.abs(d-np.median(d))))**2) if len(d) else float('nan')


def _quantum(values, n=200_000):
    v = np.unique(np.asarray(values, float).ravel()[:n])
    dv = np.diff(v)
    return float(dv[dv > 0].min()) if (dv > 0).any() else 0.


def temporal_sensor_noise(dataset, pairs=24, bins=8, max_pixels=200_000, seed=0):
    K, C = len(dataset.times), len(dataset.cameras)
    ks = sorted(set(np.linspace(0, K-2, min(pairs, K-1)).round().astype(int).tolist())) if K > 1 else []
    g = np.random.default_rng(seed)
    dI, dD, zD, depth_q = [], [], [], []
    for k in ks:
        for c in range(C):
            a, b = dataset.observation(k, c), dataset.observation(k+1, c)
            ok = np.asarray(a.mask, bool) & np.asarray(b.mask, bool) & np.asarray(a.raw_valid, bool) & np.asarray(b.raw_valid, bool)
            idx = np.flatnonzero(ok)
            if len(idx) > max_pixels: idx = np.sort(g.choice(idx, max_pixels, replace=False))
            ra, rb = np.asarray(a.rgb, float).reshape(-1, 3)[idx], np.asarray(b.rgb, float).reshape(-1, 3)[idx]
            da, db = np.asarray(a.depth, float).ravel()[idx], np.asarray(b.depth, float).ravel()[idx]
            dI.append(rb-ra); dD.append(db-da); zD.append((da+db)/2)
            if len(depth_q) < 4: depth_q.append(_quantum(da))
    if not dI: raise ValueError('noise calibration needs at least two frames')
    dI, dD, zD = np.concatenate(dI), np.concatenate(dD), np.concatenate(zD)
    q_rgb = 1/255.
    q_d = min([q for q in depth_q if q > 0], default=0.)
    rgb_var = np.maximum([_mad_var(dI[:, ch])/2 for ch in range(3)], q_rgb**2/12)
    edges = np.quantile(zD, np.linspace(0, 1, bins+1))
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (zD >= lo) & (zD <= hi)
        if m.sum() >= 100: rows.append((float(np.median(zD[m])), _mad_var(dD[m])/2, int(m.sum())))
    z, v, n = (np.array(x, float) for x in zip(*rows))
    w = np.sqrt(n)
    (a_d, b_d), _ = nnls(np.c_[np.ones_like(z), z*z]*w[:, None], v*w)
    a_d = max(float(a_d), q_d**2/12)
    return {'pairs': ks, 'pixels': int(len(dD)), 'rgb_variance': rgb_var.tolist(), 'rgb_quantum': q_rgb, 'depth_quantum_m': q_d,
            'depth_bins': [{'z_median_m': float(a), 'variance_m2': float(b), 'pixels': int(c)} for a, b, c in rows],
            'depth_variance_m2': a_d, 'depth_squared_variance': float(b_d),
            'method': 'MAD of consecutive-frame differences of pixels valid in both frames, var = (1.4826 MAD)^2 / 2; depth a + b z^2 by NNLS on depth-quantile bins'}


def _model_variance(r2, v_pred):
    """m >= 0 with median(r^2 / (v_pred + m)) = median(chi2_1) (0 if the predicted variance already explains it)."""
    target = float(chi2.ppf(.5, 1))
    f = lambda m: float(np.median(r2/(v_pred+m)))-target
    if len(r2) == 0 or f(0.) <= 0: return 0.
    lo, hi = 0., float(np.median(r2))/target+1e-30
    while f(hi) > 0: hi *= 2
    for _ in range(80):
        mid = (lo+hi)/2
        lo, hi = (mid, hi) if f(mid) > 0 else (lo, mid)
    return hi


def cross_camera_mismatch(dataset, config, policy, device, max_points=200_000, seed=0):
    from scene.endpoint import endpoint_samples
    from estimator.projective_camera import CameraRig
    from estimator.projective_sensor import MeasuredFrame, SensorNoise, CODE
    from estimator.projective_terms import build_terms
    from estimator.projective_visibility import visibility, position_sigma_z2
    pol = {**policy, 'row_innovation_gate': 'depth_front_only', 'rgb_model_variance': 0., 'depth_model_variance': 0.}
    rig = CameraRig(dataset.cameras, device); noise = SensorNoise.from_config(config, pol, device)
    sens_rgb = torch.as_tensor(np.diag(np.asarray(config.color_covariance, float)), dtype=torch.float64, device=device)
    g = np.random.default_rng(seed)
    r2_rgb, v_rgb, r2_d, v_d, per_anchor = [], [], [], [], {}
    for anchor in ('first', 'last'):
        k0, _, xyz, rgb, cov, source = endpoint_samples(dataset, anchor, config)
        sel = np.sort(g.choice(len(xyz), max_points, replace=False)) if len(xyz) > max_points else np.arange(len(xyz))
        t = lambda a: torch.as_tensor(np.asarray(a, float), dtype=torch.float64, device=device)
        x, a, S, src = t(xyz[sel]), t(rgb[sel]), t(cov[sel]), torch.as_tensor(source[sel, 1], device=device)
        frame = MeasuredFrame(dataset, k0, device)
        vreason, _ = visibility(rig.project(x, jacobian=False), position_sigma_z2(rig.R.to(S.dtype), S), frame.H, frame.W, rig.near, pol['visibility_n_sigma'])
        rows = torch.nonzero((vreason.T == CODE['VALID']) & (torch.arange(rig.count, device=device)[None, :] != src[:, None]))
        p, c = rows[:, 0], rows[:, 1]
        tm = build_terms(rig, frame, c, p, x[p], a[p], S[p], noise, pol)
        ok = tm.valid()
        r2_rgb.append((tm.r[ok, :3]**2).cpu().numpy().ravel())
        v_rgb.append((tm.var[ok, :3]+sens_rgb[None]).cpu().numpy().ravel())
        r2_d.append((tm.r[ok, 3]**2).cpu().numpy()); v_d.append(tm.var[ok, 3].cpu().numpy())
        per_anchor[anchor] = {'frame': int(k0), 'points': int(len(sel)), 'cross_camera_rows': int(len(rows)), 'valid_rows': int(ok.sum()),
                              'rgb_abs_residual_median': float(tm.r[ok, :3].abs().median()) if bool(ok.any()) else None,
                              'depth_abs_residual_median_m': float(tm.r[ok, 3].abs().median()) if bool(ok.any()) else None}
    r2_rgb, v_rgb, r2_d, v_d = (np.concatenate(x) for x in (r2_rgb, v_rgb, r2_d, v_d))
    return {'rgb_model_variance': _model_variance(r2_rgb, v_rgb), 'depth_model_variance': _model_variance(r2_d, v_d), 'anchors': per_anchor,
            'standardized_median_before': {'rgb': float(np.median(r2_rgb/v_rgb)) if len(r2_rgb) else None, 'depth': float(np.median(r2_d/v_d)) if len(r2_d) else None,
                                           'chi2_1_median': float(chi2.ppf(.5, 1))},
            'method': 'endpoint clouds sampled in every other camera at the same frame; additive model variance by median(r^2/(v_sensor+m)) = median(chi2_1) '
                      '(canonical back-projection error not credited: upper bound of the persistent mismatch)'}


def calibrate(dataset, config, policy, device, pairs=24):
    """Returns (config', policy', report): copies with the measured R; the inputs are not modified."""
    sensor = temporal_sensor_noise(dataset, pairs)
    cfg, pol = copy.deepcopy(config), copy.deepcopy(policy)
    cfg.color_covariance = np.diag(sensor['rgb_variance'])
    cfg.depth_variance, cfg.depth_squared_variance = sensor['depth_variance_m2'], sensor['depth_squared_variance']
    mismatch = cross_camera_mismatch(dataset, cfg, pol, device)
    pol['rgb_model_variance'], pol['depth_model_variance'] = mismatch['rgb_model_variance'], mismatch['depth_model_variance']
    for name, units in (('color_covariance', 'encoded RGB [0,1]^2'), ('depth_variance', 'm^2'), ('depth_squared_variance', 'dimensionless')):
        v = getattr(cfg, name)
        cfg.provenance[name] = {'source_file': str(getattr(dataset, 'root', '')), 'section_or_key': 'scene/noise_calibration.py temporal_sensor_noise', 'units': units,
                                'value': v.tolist() if isinstance(v, np.ndarray) else v, 'class': 'measured-from-scene-frames'}
    pol['sensor_noise'] = 'measured'
    report = {'sensor': sensor, 'model_mismatch': mismatch, 'ground_truth_used': False,
              'result': {'rgb_variance': sensor['rgb_variance'], 'depth_variance_m2': cfg.depth_variance, 'depth_squared_variance': cfg.depth_squared_variance,
                         'rgb_model_variance': pol['rgb_model_variance'], 'depth_model_variance_m2': pol['depth_model_variance']},
              'replaced_uncalibrated': {'rgb_variance': np.diag(np.asarray(config.color_covariance, float)).tolist(), 'depth_variance_m2': config.depth_variance,
                                        'depth_squared_variance': config.depth_squared_variance, 'rgb_model_variance': policy['rgb_model_variance'],
                                        'depth_model_variance_m2': policy['depth_model_variance']}}
    return cfg, pol, report


def velocity_prior(speed99, dims=3):
    """Isotropic P_vv from a DECLARED physical speed v_99 enclosing 99% of the expected speeds (audit item 8):
    |v| ~ sigma_v chi_3, P(|v| <= v_99) = 0.99  ->  sigma_v = v_99 / sqrt(chi2_3(0.99)) = v_99 / 3.368."""
    if not (np.isfinite(speed99) and speed99 > 0): raise ValueError('--velocity-prior-speed99 must be a positive finite speed (m/s)')
    s = speed99/np.sqrt(chi2.ppf(.99, dims))
    return s*s*np.eye(dims), float(s)
