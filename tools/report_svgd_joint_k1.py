#!/usr/bin/env python3
"""Assemble reports/svgd_joint_k1/ (spec section 10) from a finished batch folder and the K = 1 unit tests.

    python -B tools/report_svgd_joint_k1.py --batch BATCH_ROOT [--run-tests]

Writes implementation_and_validation.md, equation_to_code_contract.json, calibration_provenance.json, theory_deviation_register.json,
unit_test_results.json, optimizer_diagnostics.json, posterior_calibration.json, performance.json, run_manifest.json and root_cause.md
(one section per failed gate). Nothing in the batch folder is modified.
"""
import argparse
import glob
import json
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUT = REPO/'reports/svgd_joint_k1'

CONTRACT = [
    ('spec 2', 'T_0 = I, T_t = T_{t-1} Exp(dt nu_t)', 'estimator/svgd_k1_motion.py:se3_exp, compose', 'test_svgd_k1_math.test_se3_exp_log_and_finite_differences', 'EXACT'),
    ('spec 2', 'nu random-walk prior paid once per body', 'estimator/svgd_k1_motion.py:MotionPrior.log_prob', 'test_svgd_k1_math.test_motion_prior_normalized_and_paid_once', 'EXACT'),
    ('spec 3.1', 'normalized observed XYZ-RGB KDE, Lambda = sensor + representation', 'estimator/svgd_k1_observation.py:FrameKDE, representation_scales', 'test_svgd_k1_math.test_kde_potential_dense_oracle_and_certificate', 'EXACT within certificate'),
    ('spec 3.2', 'b_0 frozen reference Gaussian; ell = rho + (1-rho) p/b', 'estimator/svgd_k1_observation.py:Background, log_potential', 'test_svgd_k1_math.test_kde_potential_dense_oracle_and_certificate', 'EXACT'),
    ('spec 3.2', 'rho_t by held-out spatial-block CV', 'estimator/svgd_k1_observation.py:calibrate_rho', 'test_svgd_k1_runner.test_calibration_reproducible_and_no_silent_floor', 'DECLARED MODEL CHOICE'),
    ('spec 8.4', 'certified spatial candidate grid', 'estimator/svgd_k1_observation.py:Grid, choose_kappa (CSR pairs)', 'test_svgd_k1_math.test_kde_potential_dense_oracle_and_certificate', 'NUMERICAL_APPROX (certified bound)'),
    ('spec 3.3', 'shared 6D anchor integral once', 'estimator/svgd_k1_reference.py:laplace_at_mean, qmc_log_evidence', 'test_svgd_k1_math.test_shared_anchor_integral_once_vs_per_frame', 'EXACT on Gaussian toy'),
    ('spec 3.3', 'Laplace at prior mean + CRN-QMC fallback', 'estimator/svgd_k1_reference.py; estimator/svgd_k1_joint_posterior.py', 'test_svgd_k1_math.test_laplace_vs_qmc_on_kde_scene', 'NUMERICAL_APPROX (measured)'),
    ('spec 3.3', 'a_eff = n_eff (1/rho_i)/sum(1/rho_k)', 'estimator/svgd_k1_run.py:build_model; svgd_k1_observation.kernel_sum', 'test_svgd_k1_math.test_a_eff_duplication_invariance', 'DECLARED MODEL CHOICE'),
    ('spec 4', 'F with exactly summed labels; gamma; dF/dnu; dF/dh', 'estimator/svgd_k1_joint_posterior.py:JointPosterior.evaluate', 'test_svgd_k1_math.test_enumeration_gradients_identity_and_no_evidence', 'EXACT (2^N, FD)'),
    ('spec 5', 'SVGD over motion hypotheses with grad F; repulsion sign', 'estimator/svgd_k1_stein.py:svgd_direction', 'test_svgd_k1_math.test_svgd_direction_uses_score_and_repulsion_sign', 'EXACT'),
    ('spec 1.1', 'point-only boundary; GT-leak sentinel', 'scene/svgd_k1_cloud_adapter.py', 'test_svgd_k1_math.test_point_only_boundary_and_gt_leak', 'ENFORCED'),
    ('spec 9 G2', 'camera/point order invariance; duplication bounded', 'estimator/svgd_k1_run.py:build_model', 'test_svgd_k1_runner.*', 'TESTED'),
]
DEVIATIONS = [
    {'id': 'K1-A01', 'classification': 'NUMERICAL_APPROX', 'what': 'Laplace at the PRIOR MEAN (second-order expansion), not at the integrand mode',
     'measured': 'B error median ~0.1 nat, max ~0.7 nat when reference sigma ~ KDE bandwidth (8192-node QMC, synthetic); exact for Gaussian factors',
     'control': 'reference.integration = crn_qmc (validated higher-accuracy route, S x cost)'},
    {'id': 'K1-A02', 'classification': 'NUMERICAL_APPROX', 'what': 'Laplace validity margin lambda_min(M) >= 0.5 -> CRN-QMC (64 nodes) for the remaining points',
     'measured': 'near-singular M produced corrections up to +590 nats on Blade before the margin; fallback counts logged per evaluation'},
    {'id': 'K1-A03', 'classification': 'NUMERICAL_APPROX (certified)', 'what': 'spatial candidate grid of radius kappa sqrt(lambda_max(LX)) with per-query bound on log ell',
     'measured': 'certificate tolerance 1e-6 nat per point-frame on the declared domain; out-of-domain queries counted'},
    {'id': 'K1-M01', 'classification': 'DECLARED MODEL CHOICE', 'what': 'rho_t from held-out spatial-block CV (block = 8 x KDE spacing, 4 folds, Beta(1,1))', 'measured': 'per-frame folds and IQR in calibration_provenance.json'},
    {'id': 'K1-M02', 'classification': 'DECLARED MODEL CHOICE', 'what': 'a_eff composite-likelihood weights with H_X = s_0^2 I, H_C = Lambda_C(frame 0)', 'measured': 'duplication invariance tested'},
    {'id': 'K1-M03', 'classification': 'DECLARED ASSUMPTION (NOT CALIBRATED)', 'what': 'rotation prior sd 2 rad/s, rotation random walk 1 rad^2/s^3; translation from the repository noise file'},
    {'id': 'K1-H01', 'classification': 'HARDWARE BUDGET', 'what': 'overnight schedule: 4 particles x 12 SVGD iterations, 12 MAP iterations from two starts; prefixes of 10/4/2 frames',
     'measured': '~4 s per particle-frame at 34k points on the shared 8 GB RTX 4070; full sequences infeasible in one night'},
    {'id': 'K1-C01', 'classification': 'CONTROL', 'what': 'MAP of the same F from identity and from the best SVGD particle; MAP-gamma seeds reported as control only'},
]


def run_tests():
    r = subprocess.run([sys.executable, '-B', '-m', 'unittest', 'tests.test_svgd_k1_math', 'tests.test_svgd_k1_runner', '-v'], cwd=REPO, capture_output=True, text=True,
                       env={'CUDA_VISIBLE_DEVICES': '', 'PATH': '/usr/bin:/bin'})
    lines = [l for l in r.stderr.splitlines() if l.endswith(('ok', 'FAIL', 'ERROR')) or ' ... ' in l]
    return {'returncode': r.returncode, 'summary': r.stderr.strip().splitlines()[-1] if r.stderr.strip() else '', 'tests': lines}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--batch', type=Path, required=True); ap.add_argument('--run-tests', action='store_true'); a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True); b = json.loads((a.batch/'batch_status.json').read_text())
    jobs, cal, opt, perf, man, oracles, roots = [], {}, {}, {}, {}, {}, []
    for j in b['jobs']:
        run = Path(j['output']); sv = run/'svgd_joint_k1'
        ld = lambda p: json.loads(p.read_text()) if p.is_file() else None
        st, c, o, pf, m = ld(sv/'status.json'), ld(sv/'calibration_provenance.json'), ld(sv/'optimizer_diagnostics.json'), ld(sv/'performance.json'), ld(run/'run_manifest.json')
        orc = sorted(glob.glob(str(run/'posthoc/k1_oracle*/oracle.json'))); orc = json.loads(Path(orc[-1]).read_text()) if orc else None
        cal[j['name']] = c; opt[j['name']] = o; perf[j['name']] = pf; man[j['name']] = m; oracles[j['name']] = orc
        jobs.append({'name': j['name'], 'state': j['state'], 'seconds': j.get('seconds'), 'stage': st and st.get('stage'), 'F_identity': st and st.get('F_identity'),
                     'F_svgd_best': st and st.get('F_svgd_best'), 'F_map': st and st.get('F_map'), 'seeds': st and st.get('seeds'), 'map_seeds': st and st.get('map_seeds'),
                     'pi': st and st.get('pi_posterior_mean'), 'oracle': orc and orc['status'], 'oracle_F': orc and orc['F'], 'oracle_seeds': orc and orc['seeds'],
                     'oracle_map_seeds': orc and orc.get('map_seeds_control'), 'pose_error_map': orc and orc.get('pose_error_map'),
                     'rho_median': c and sorted(r['rho'] for r in c['rho'])[len(c['rho'])//2], 'grid_radius_mm': c and 1e3*sorted(c['grid_radius_m'])[len(c['grid_radius_m'])//2],
                     'wall_s': pf and pf.get('wall_seconds'), 'peak_gb': pf and (pf.get('peak_allocated_bytes') or 0)/1e9})
        if orc and not str(orc['status']).startswith('PASS'): roots.append((j['name'], orc))
        if st and st.get('stage') not in ('DONE',): roots.append((j['name'], {'status': f"RUN {st.get('stage')}", 'failure': st.get('failure')}))
    w = lambda n, x: (OUT/n).write_text(json.dumps(x, indent=1, default=str))
    w('equation_to_code_contract.json', [dict(zip(('spec', 'equation', 'code', 'test', 'status'), r)) for r in CONTRACT]); w('theory_deviation_register.json', DEVIATIONS)
    w('calibration_provenance.json', cal); w('optimizer_diagnostics.json', opt); w('performance.json', perf); w('run_manifest.json', {'batch': str(a.batch), 'batch_status': b, 'runs': man})
    w('posterior_calibration.json', {'status': 'NOT_RUN', 'probability_semantics': 'UNCALIBRATED_APPROXIMATION',
                                     'reason': 'no held-out synthetic reliability study and no low-dimensional exact pose-posterior comparison (Gate 4) were run in this batch'})
    tests = run_tests() if a.run_tests else {'status': 'NOT_RUN_BY_THIS_INVOCATION'}; w('unit_test_results.json', tests)
    L = ['# Root causes of failed gates (joint K = 1)', '']
    for name, o in roots: L += [f'## {name}: {o["status"]}', '', '```', json.dumps({k: o.get(k) for k in ('F', 'seeds', 'map_seeds_control', 'pose_error_map', 'gamma_at_gt_pose', 'B_at_gt_pose', 'failure') if k in o}, indent=1, default=str), '```', '']
    (OUT/'root_cause.md').write_text('\n'.join(L))
    H = ['# Joint K = 1 static/moving support + shared rigid motion (SVGD over motion hypotheses) — implementation and validation', '',
         f"Generated {time.strftime('%Y-%m-%d %H:%M')} from `{a.batch}`. Spec: `CODEX_RGBDiekf_SVGD_JOINT_RIGID_SUPPORT_K1_2026-11-03.md`.", '',
         '| job | state | F identity | F SVGD best | F MAP | seeds (SVGD) | MAP-control seeds | oracle | GT − identity | best − GT | seed precision / recall | MAP-control precision / recall | rho | wall |',
         '|---|---|---|---|---|---|---|---|---|---|---|---|---|---|']
    f = lambda x, p=1: '—' if x is None else (f'{x:.{p}f}' if isinstance(x, float) else str(x))
    for r in jobs:
        of = r['oracle_F'] or {}; os_ = r['oracle_seeds'] or {}; om = r['oracle_map_seeds'] or {}
        H.append(f"| {r['name']} | {r['state']} | {f(r['F_identity'])} | {f(r['F_svgd_best'])} | {f(r['F_map'])} | {f(r['seeds'])} | {f(r['map_seeds'])} | {r['oracle'] or '—'} | "
                 f"{f(of.get('gt_minus_identity'))} | {f(of.get('best_found_minus_gt'))} | {f(os_.get('precision'), 3)} / {f(os_.get('recall'), 3)} | {f(om.get('precision'), 3)} / {f(om.get('recall'), 3)} | "
                 f"{f(r['rho_median'], 3)} | {f(r['wall_s'], 0)} s |")
    H += ['', 'Probabilities are `UNCALIBRATED_APPROXIMATION` (Gate 4 not run). Prefix runs are `PREFIX_DIAGNOSTIC`. GT only in the post-hoc oracle.', '',
          'See `equation_to_code_contract.json`, `theory_deviation_register.json`, `root_cause.md`, `unit_test_results.json`.', '']
    (OUT/'implementation_and_validation.md').write_text('\n'.join(H)); print('\n'.join(H))


if __name__ == '__main__':
    main()
