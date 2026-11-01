#!/usr/bin/env python3
"""Measured numbers behind the Phase-1 gate tests (tests/test_temporal_phase1_*.py assert; this prints what they observe), for
reports/temporal_phase1_ieks.md. Analytic fixtures only (tests/temporal_fixtures.py); ground truth used for evaluation only.

    python tools/temporal_phase1_gate_evidence.py OUT.json
"""
import json
import math
import sys
from pathlib import Path
import numpy as np
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from tests.temporal_fixtures import (fronto_cameras, AffinePlate, AnalyticDataset, catalogue, run_solver, dense_oracle_map, nees, same_map,  # noqa: E402
                                     run_original_filter)
from tests.test_temporal_phase1_static import RING, interior_pixels  # noqa: E402
from tests.test_temporal_phase1_motion import MOTIONS, QA, TIMES, true_traj  # noqa: E402


def g22():
    out = []
    for src, box in ((6, (25, 70, 20, 52)), (0, (10, 40, 10, 30))):
        for direction in np.eye(3).tolist()+[[1, -1, 1]]:
            pix = interior_pixels(12, seed=src, box=box)
            ds = AnalyticDataset(fronto_cameras(RING), [AffinePlate(1.5, (-.8, .8), (-.6, .6))], np.arange(20)/30.)
            cat = catalogue(ds, src, pix); sd = np.sqrt(np.diag(cat['P_x0'][0]))
            delta = .8*sd*np.asarray(direction, float)/np.linalg.norm(direction); cat['x_ref'] = cat['x_ref']+delta
            res = run_solver(ds, cat)
            ok = res['trajectory_status'].numpy() != 4
            b, bc = res['b'].numpy(), res['b_cov'].numpy()
            out.append({'source_camera': src, 'direction': direction, 'delta_mm': (1e3*delta).tolist(),
                        'recovery_ratio_median': float(np.median(np.linalg.norm(b[ok]+delta, axis=1)/np.linalg.norm(delta))),
                        'nees_b_max': float(max(nees(b[i]+delta, bc[i]) for i in np.flatnonzero(ok))),
                        'max_abs_velocity_m_s': float(np.abs(res['velocity'].numpy()[:, ok]).max())})
    return out


def g31_32():
    rows = {}
    for name, motion in MOTIONS.items():
        for delta in (np.zeros(3), np.array([1., -1., 1.])*1.6e-3):
            ds = AnalyticDataset(fronto_cameras(RING), [AffinePlate(1.5, (-.8, .8), (-.6, .6), motion=motion)], TIMES)
            cat = catalogue(ds, 6, interior_pixels(20, seed=5)); x0 = cat['x_ref'].copy(); cat['x_ref'] = x0+delta
            res = run_solver(ds, cat, qa=QA[name]); old = run_original_filter(ds, cat, qa=QA[name])
            ok = res['trajectory_status'].numpy() != 4
            gt = true_traj(motion, x0, TIMES)
            e = np.linalg.norm(res['smooth_mean'].numpy()[:, ok, :3]-gt[:, ok], axis=-1); eo = np.linalg.norm(old[:, ok, :3]-gt[:, ok], axis=-1)
            mot = np.linalg.norm(gt[-1, ok]-gt[0, ok], axis=-1)
            support = [[c for c in range(len(ds.cameras)) if res['camera_reason'][k, c, 0] == 0] for k in range(len(TIMES))]
            So, Co = dense_oracle_map(ds, cat, 0, support, res['warm_start_smooth_mean'].numpy()[:, 0], qa=QA[name])
            rows[f'{name}{"+delta" if delta.any() else ""}'] = {
                'q_a': QA[name], 'true_motion_mm_median': float(1e3*np.median(mot)), 'new_error_mm_median_all_frames': float(1e3*np.median(e)),
                'new_final_error_mm_median': float(1e3*np.median(e[-1])), 'old_error_mm_median_all_frames': float(1e3*np.median(eo)),
                'old_final_error_mm_median': float(1e3*np.median(eo[-1])), 'new_frame0_error_mm_median': float(1e3*np.median(e[0])),
                'old_frame0_error_mm_median': float(1e3*np.median(eo[0])), 'oracle_squared_mahalanobis_point0': same_map(res['smooth_mean'].numpy()[:, 0], So, Co),
                'b_recovery_ratio_median': float(np.median(np.linalg.norm(res['b'].numpy()[ok]+delta, axis=1)/np.linalg.norm(delta))) if delta.any() else None,
                'trajectory_status_counts': np.bincount(res['trajectory_status'].numpy(), minlength=6).tolist()}
    return rows


def g23_g44(mode):
    sig = (.01, 1e-3)
    out = {}
    for GX in ((1., 30., 300.) if mode == 'coverage' else (300., 3000.)):
        plate = lambda: [AffinePlate(1.5, (-.8, .8), (-.6, .6), c0=(.5, .5, .5), Gx=(GX*.8, -GX*.3, GX*.2), Gy=(GX*.1, GX*.6, -GX*.5))]
        pix = interior_pixels(20, seed=9)
        clean = AnalyticDataset(fronto_cameras(RING), plate(), np.arange(10)/30.); base = catalogue(clean, 6, pix); truth = base['x_ref']
        for variant in (('stage_a',) if mode == 'coverage' else ('naive', 'latent')):
            N, tot, exc = [], 0, 0
            for run in range(6):
                ds = AnalyticDataset(fronto_cameras(RING), plate(), np.arange(10)/30., noise=sig, seed=run)
                if mode == 'coverage':
                    g = np.random.default_rng(100+run); cat = dict(base)
                    cat['x_ref'] = truth+np.stack([g.multivariate_normal(np.zeros(3), P) for P in base['P_x0']])
                    res = run_solver(ds, cat, noise_sig=sig)
                else:
                    cat = catalogue(ds, 6, pix, depth_var=sig[1]**2)
                    res = run_solver(ds, cat, noise_sig=sig, latent_colour=(variant == 'latent'))
                ok = res['trajectory_status'].numpy() == 0; tot += len(ok); exc += int((~ok).sum())
                b, bc = res['b'].numpy(), res['b_cov'].numpy()
                N += [nees(cat['x_ref'][i]+b[i]-truth[i], bc[i]) for i in np.flatnonzero(ok)]
            out[f'gradient_per_px_{GX*.01:g}_{variant}'] = {'mean_nees_b_converged': float(np.mean(N)), 'samples': len(N), 'excluded_not_converged_fraction': exc/tot}
    return out


def g5():
    from tests.test_temporal_phase1_identification import simulate, TRUE
    from estimator import temporal_model_identification as TI
    tracks, _ = simulate(300, 25, 0)
    o = TI.em(tracks, (1., 1., 1e-4), iters=400)
    th = tuple(o['theta'][n] for n in TI.NAMES); idf = TI.identifiability(tracks, th)
    short, _ = simulate(1, 3, 6, missing=0.)
    o2 = TI.em(short, (1., 1., 1e-4), iters=200); idf2 = TI.identifiability(short, tuple(o2['theta'][n] for n in TI.NAMES))
    times = np.arange(25)*.033; tr3, s = simulate(300, 25, 1, missing=0., times=times)
    naive = float(np.mean([TI.finite_difference_q(tr3['y'].numpy()[:, i], times) for i in range(300)]))
    return {'true': dict(zip(TI.NAMES, TRUE)), 'em_estimate': o['theta'], 'se_log': dict(zip(TI.NAMES, idf['se_log'])), 'status': idf['status'], 'em_iterations': o['iterations'],
            'short_sequence_status': idf2['status'], 'finite_difference_q_naive': naive, 'em_q_same_data': TI.em(tr3, (1., 1., 1e-4), iters=400)['theta']['q']}


def main():
    out = Path(sys.argv[1])
    if out.exists(): raise SystemExit(f'{out} exists')
    ev = {'G2_2': g22(), 'G3_1_G3_2': g31_32(), 'G2_3_coverage_vs_texture': g23_g44('coverage'), 'G4_4_naive_vs_latent': g23_g44('latent'), 'G5': g5()}
    out.write_text(json.dumps(ev, indent=1, default=float))
    print(json.dumps(ev, default=float)[:6000])


if __name__ == '__main__':
    main()
