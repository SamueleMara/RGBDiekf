#!/usr/bin/env python3
"""Theory-parity audit of the SVGD Stage 1 (spec TF.3-TF.7): runs tests/test_svgd_*.py in-process, collects the measured
discrepancies recorded by the tests (tests/svgd_fixtures.WITNESS), and regenerates from estimator/svgd_contract.py:
    docs/SVGD_THEORY_TO_CODE_CONTRACT.md            equation-by-equation contract + traceability matrix
    reports/svgd/theory_parity_summary.json         per-equation gate outcome, discrepancies, source hashes, evidence
    reports/svgd/theory_deviation_register.{md,json}

    python -B tools/svgd_theory_parity.py            # full audit (CUDA tests run when a GPU is visible)
"""
import hashlib
import io
import json
import subprocess
import sys
import time
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from estimator import svgd_contract as SC

GATES = ('PASS_THEORY_EXACT', 'PASS_AUTHORIZED_APPROXIMATION', 'FAILED_THEORY_MISMATCH', 'BLOCKED_MODEL_UNSPECIFIED', 'BLOCKED_THEORY_CONFLICT',
         'BLOCKED_EXACT_RUNTIME', 'BLOCKED_THEORY_FIDELITY', 'NOT_TESTED')


FAILURE_REVEALING = {'occluded_static_behind_blade', 'camera_duplication_bias', 'bias_sign_undetermined', 'textured_vs_flat_tangential_information', 'occam'}
TOY_CRITERION = ('toy reporting criterion (declared, GT post hoc only): static drift median <= 0.5 mm (half the fixture 1 mm depth sigma); '
                 'moving recovered-fraction median >= 0.9 with the same static criterion')


def physical_outcome(g):
    if 'static_drift_median_m' not in g: return g.get('status', 'NOT_RUN')
    static_ok = g['static_drift_median_m'] <= 5e-4
    if not g.get('moving_true_displacement_median_m'): return 'NO_FALSE_DRIFT' if static_ok else 'FALSE_DRIFT_FAILED'
    return 'RECOVERED' if static_ok and (g.get('moving_recovered_fraction_median') or 0) >= .9 else 'PHYSICAL_MOTION_NOT_RECOVERED'


def sha(p): return hashlib.sha256((REPO/p).read_bytes()).hexdigest() if (REPO/p).is_file() else None


def run_tests():
    loader = unittest.TestLoader(); suite = loader.discover(str(REPO/'tests'), pattern='test_svgd_*.py', top_level_dir=str(REPO))
    outcome = {}
    class R(unittest.TextTestResult):
        def addSuccess(s, t): super().addSuccess(t); outcome[t.id()] = 'PASS'
        def addFailure(s, t, e): super().addFailure(t, e); outcome[t.id()] = 'FAIL'
        def addError(s, t, e): super().addError(t, e); outcome[t.id()] = 'ERROR'
        def addSkip(s, t, r): super().addSkip(t, r); outcome[t.id()] = f'SKIPPED: {r}'
    buf = io.StringIO(); t0 = time.time()
    res = unittest.TextTestRunner(stream=buf, resultclass=R, verbosity=1).run(suite)
    from tests.svgd_fixtures import WITNESS
    return outcome, WITNESS, {'ran': res.testsRun, 'failures': len(res.failures), 'errors': len(res.errors), 'skipped': len(res.skipped), 'seconds': time.time()-t0}, buf.getvalue()


def test_status(outcome, tid):
    hits = {k: v for k, v in outcome.items() if k == 'tests.'+tid.split('tests/')[-1].replace('/', '.') or k.startswith('tests.'+tid.split('tests/')[-1]+'.')}
    if not hits: return 'NOT_RUN'
    if any(v in ('FAIL', 'ERROR') for v in hits.values()): return 'FAIL'
    return 'PASS' if all(v == 'PASS' for v in hits.values()) else 'SKIPPED'


def gate(eq, statuses):
    dev = next((d for d in SC.DEVIATIONS if d['id'] == eq['deviation']), None)
    if eq['classification'] in ('NOT_DEFINED_IN_THEORY', 'BLOCKED_MODEL_UNSPECIFIED'): return 'BLOCKED_MODEL_UNSPECIFIED'
    if eq['classification'] == 'OUT_OF_SCOPE_PHASE1': return 'NOT_TESTED'
    if eq['deviation'].startswith('TC-'): return 'BLOCKED_THEORY_CONFLICT'
    if not statuses or all(s == 'NOT_RUN' for s in statuses): return 'NOT_TESTED'
    if any(s == 'FAIL' for s in statuses): return 'FAILED_THEORY_MISMATCH'
    if eq['classification'] == 'EXACT_MODEL' and dev is None: return 'PASS_THEORY_EXACT' if all(s == 'PASS' for s in statuses) else 'NOT_TESTED'
    if dev is None and eq['classification'] in ('MODEL_APPROX', 'NUMERICAL_APPROX') and all(s == 'PASS' for s in statuses):
        return 'PASS_AUTHORIZED_APPROXIMATION'                     # approximation named by the spec itself (e.g. Laplace, spec E4)
    if dev is not None and dev['authorization'].get('decision', '').startswith('AUTHORIZED'): return 'PASS_AUTHORIZED_APPROXIMATION'
    return 'BLOCKED_THEORY_FIDELITY'


def main():
    commit = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=REPO, capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(['git', 'status', '--short'], cwd=REPO, capture_output=True, text=True).stdout.splitlines()
    outcome, wit, summary, log = run_tests()
    rows = []
    for eq in SC.EQUATIONS:
        st = [test_status(outcome, t) for t in eq['tests']]
        files = sorted({eq['function'].split(':')[0]} | {t.split('.')[0]+'.py' for t in eq['tests']})
        meas = [{**m, 'test_semantics': 'FAILURE_REVEALING: PASS = the failure mode was detected, NOT that recovery passed'} if m.get('test') in FAILURE_REVEALING else m
                for m in wit.get(eq['id'], [])]
        rows.append({**eq, 'test_status': dict(zip(eq['tests'], st)), 'gate': gate(eq, st), 'measured': meas,
                     'source_sha256': {f: sha(f) for f in files if sha(f)}, 'cuda': 'PASS' if any(m.get('test', '').startswith('cuda_parity') for m in wit.get(eq['id'], [])) else 'NOT_RUN',
                     'prohibited_substitution': SC.PROHIBITED.get(eq['id'], 'any change without a deviation record')})
    counts = {'exact_model_operators': sum(r['gate'] == 'PASS_THEORY_EXACT' for r in rows), 'verified_equivalent_optimizations': len(SC.EQUIVALENT_OPTIMIZATIONS),
              'controlled_numerical_approximations': sum(d['classification'] == 'NUMERICAL_APPROX' for d in SC.DEVIATIONS),
              'model_approximations': sum(d['classification'] == 'MODEL_APPROX' for d in SC.DEVIATIONS),
              'non_equivalent_deviations': sum(d['conclusion'] == 'NON_EQUIVALENT' for d in SC.DEVIATIONS),
              'unapproved_deviations': sum(not d['authorization']['decision'].startswith('AUTHORIZED') for d in SC.DEVIATIONS),
              'blocked_equations': sum(r['gate'].startswith('BLOCKED') for r in rows), 'untested_equations': sum(r['gate'] == 'NOT_TESTED' for r in rows),
              'failed_equations': sum(r['gate'] == 'FAILED_THEORY_MISMATCH' for r in rows)}
    overall = 'FAILED_THEORY_MISMATCH' if counts['failed_equations'] else 'BLOCKED_MODEL_UNSPECIFIED' if counts['blocked_equations'] else 'PASS_THEORY_EXACT'
    out = {'generated': time.strftime('%Y-%m-%d %H:%M:%S'), 'commit': commit, 'working_tree_dirty': dirty, 'theory_pdf_sha256': sha(SC.THEORY_PDF),
           'spec_sha256': sha(SC.SPEC), 'contract_sha256': sha('estimator/svgd_contract.py'), 'label_mapping': SC.LABEL_MAPPING,
           'overall_theory_fidelity': overall, 'counts': counts, 'test_summary': summary, 'test_outcomes': outcome, 'equations': rows,
           'equivalent_optimizations': SC.EQUIVALENT_OPTIMIZATIONS, 'deviation_ids': [d['id'] for d in SC.DEVIATIONS],
           'theory_conflicts': SC.THEORY_CONFLICTS, 'gate4_criterion': TOY_CRITERION,
           'gate4_measurements': [{**g, 'physical_outcome': physical_outcome(g)} for g in wit.get('GATE4', [])],
           'physical_motion': 'PHYSICAL_MOTION_NOT_RECOVERED (Gate-4 toys; no real-scene acceptance run)',
           'scope_of_operator_parity': 'algebra and small fixtures only: operator parity is NOT end-to-end model fidelity, CUDA parity is NOT workload acceptance',
           'end_to_end_model_fidelity': 'NOT_ESTABLISHED', 'workload_acceptance': 'NOT_ESTABLISHED (synthetic 1k/10k/34k operator sizes + a 2-frame Blade smoke only)',
           'legacy_baseline': 'observation only (7 failures at d0b470b); see reports/svgd/preflight_audit_2026-11-03.md for the source-matched rerun',
           'audit_b62c7cee_findings': {k: {'status': v[0], 'response': v[1]} for k, v in SC.AUDIT_FINDINGS.items()}, 'posthoc_diagnostics': wit.get('F6', []),
           'suitability_for_joint_recovery': 'NOT_SUITABLE: Stage-1 model-evidence gate BLOCKED'}
    (REPO/'reports/svgd').mkdir(parents=True, exist_ok=True)
    clean = lambda o: json.loads(json.dumps(o, default=lambda v: v.tolist() if hasattr(v, 'tolist') else str(v)))
    (REPO/'reports/svgd/theory_parity_summary.json').write_text(json.dumps(clean(out), indent=1))
    (REPO/'reports/svgd/theory_deviation_register.json').write_text(json.dumps(SC.DEVIATIONS+[{'id': c['id'], 'kind': 'THEORY_CONFLICT', **c} for c in SC.THEORY_CONFLICTS], indent=1))
    write_register_md(); write_contract_md(rows, counts, overall, commit, summary)
    print(f"[parity] {summary}  overall {overall}  counts {counts}")
    if summary['failures'] or summary['errors']: print(log[-4000:])
    return 0 if not (summary['failures'] or summary['errors']) else 1


def write_register_md():
    L = ['# SVGD theory deviation register (append-only)', '',
         '| # | date | entry |', '|---|---|---|',
         f'| 0 | 2026-11-03 | `NO_DEVIATIONS_RECORDED` (initial entry at Gate TF0; **superseded** by entries 1-{len(SC.DEVIATIONS)+1} below) |']
    for n, d in enumerate(SC.DEVIATIONS, 1): L.append(f"| {n} | 2026-11-03 | **{d['id']}** {d['classification']} / {d['status']} / {d['conclusion']} |")
    L.append(f"| {len(SC.DEVIATIONS)+1} | 2026-11-03 | **TC-0001** THEORY_CONFLICT / OPEN_REQUIRES_USER_DECISION |")
    L.append(f"| {len(SC.DEVIATIONS)+2} | 2026-11-03 | **DEV-0007 amended** (audit b62c7cee F7): 'conservative' withdrawn -> bias sign UNDETERMINED, conclusion UNDETERMINED |")
    L += ['', 'Authorization: none of the entries is authorized; each is reachable only with `--svgd-theory-fidelity experimental` and is excluded from',
          'theory-fidelity acceptance and from any advertised probability (spec TF.4).', '']
    for d in SC.DEVIATIONS:
        L += [f"## {d['id']} — {d['classification']} ({d['status']}, {d['conclusion']})", '']
        for k in ('original_theory', 'actual_implementation', 'reason', 'mathematical_difference', 'affected_outputs', 'error_or_bias', 'independent_oracle',
                  'correctness_risk', 'alternatives_considered', 'ablation_command', 'authorization'):
            v = d[k]; L.append(f"- **{k}**: " + (', '.join(f'{a}: `{b}`' for a, b in v.items()) if isinstance(v, dict) else str(v)))
        L.append('')
    for c in SC.THEORY_CONFLICTS:
        L += [f"## {c['id']} — THEORY_CONFLICT ({c['status']})", '', f"- **theory**: {c['theory']}", f"- **spec**: {c['spec']}",
              f"- **consequence**: {c['consequence']}", f"- **implementation**: {c['implementation']}", '']
    (REPO/'reports/svgd/theory_deviation_register.md').write_text('\n'.join(L)+'\n')


def fmt_measured(ms):
    keep = []
    for m in ms:
        nums = {k: v for k, v in m.items() if isinstance(v, (int, float)) and k not in ('N',)}
        if nums: keep.append(', '.join(f'{k}={v:.2e}' if isinstance(v, float) else f'{k}={v}' for k, v in list(nums.items())[:3]))
    return '; '.join(keep[:3]) or '—'


def write_contract_md(rows, counts, overall, commit, summary):
    L = ['# SVGD Stage-1 theory-to-code contract (Gate TF0, frozen)', '',
         f'Generated by `tools/svgd_theory_parity.py` from `estimator/svgd_contract.py` (single source of truth). Checkout `{commit}` + working tree.',
         f'Theory: `{SC.THEORY_PDF}` (sha256 `{sha(SC.THEORY_PDF)[:16]}…`); spec: `{SC.SPEC}`.',
         f'The `.tex` is not in the repository: the spec\'s LaTeX labels are mapped to the PDF\'s printed equation numbers by content (`{SC.LABEL_MAPPING}`).', '',
         '## 1. Selected model (what is exact, what is proposed, what is missing)', '',
         '- **Joint target (PDF (29), E1)**: p(X, m, Z, V, B | Q_0:T) ∝ p(Q | X, V, B) p(X | m, Z) p(V, B, m, Z). Only the factors marked EXACT_MODEL below are '
         'implemented as the theory states them; p(V | X, C), B_t, p(m, Z), the moving-motion prior, Γ_t and Ω_model are **NOT_DEFINED_IN_THEORY**.',
         '- **Observation support**: every mask-positive valid-depth pixel of every camera (`scene/pointcloud_sequence.py`), world XYZ [m], encoded RGB in [0,1] as loaded; '
         'no background, no subsampling of inference points, frames or cameras; source triple (frame id, camera, pixel) unique.',
         '- **Sample weighting**: w_j = 1/M_t (PDF (8)); predicted α_i = 1/|V_t| (PDF (18)) on a V_t that the theory does not define.',
         '- **Shared uncertainty**: the frame-0 back-projection covariance Σ_i^0 enters ONCE, in P_i0 of the temporal prior (C4); the rigidity covariance counts the '
         'shared reference once (rank-1 term c 11ᵀ).',
         '- **Proposal vs likelihood**: the KDE score g_t (E03) and the Stein field Φ (E07) are **proposals only**; the single statistical observation is the feature '
         'likelihood E_F,t (E12). The score is never added to the objective; Φ is never assimilated.',
         '- **Whole sequence**: one objective over all frames, E = Σ_t E_F,t(X_t) + E_time(S) (E32, conditional); later frames revise earlier ones through the '
         'coupled temporal prior (block-tridiagonal L solved exactly).',
         '- **Probabilities**: no γ_i is published: E25/E26 are BLOCKED_MODEL_UNSPECIFIED (Gate 6). The tiny-model evidence machinery (E27-E31, E33) is exact and tested.', '',
         '## 2. Classification and traceability matrix', '',
         f'Overall: **{overall}** — counts: ' + ', '.join(f'{k} {v}' for k, v in counts.items()) + f'. Tests: {summary}.', '',
         '| ID | theory | spec | role | cat | classification | invariant | function | tests | dtype | units | measured (max) | CUDA | gate | deviation | prohibited substitution |',
         '|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|']
    for r in rows:
        tests = '<br>'.join(f"`{t.split('test_svgd_')[-1]}` {s}" for t, s in r['test_status'].items()) or '—'
        L.append(f"| {r['id']} | `{r['label']}` PDF {r['pdf']} | {r['spec']} | {r['role']} | {r['category']} | {r['classification']} | {r['invariant']} | "
                 f"`{r['function']}` | {tests} | {r['dtype']} | {r['units']} | {fmt_measured(r['measured'])} | {r['cuda']} | **{r['gate']}** | {r['deviation']} | {r['prohibited_substitution']} |")
    L += ['', 'Source SHA-256 per row, the full measured discrepancies and the tolerance bases are in `reports/svgd/theory_parity_summary.json`.', '',
          '## 3. Parameters: provenance or explicit gap', '', '| parameter | strict config | experimental config |', '|---|---|---|']
    strict = json.loads((REPO/'configs/svgd_support.json').read_text()); exp = json.loads((REPO/'configs/svgd_support_experimental.json').read_text())
    desc = lambda p: '—' if not isinstance(p, dict) else (f"`{p['class']}`" + (f" {p.get('deviation')}" if p.get('deviation') not in (None, 'NONE') else '') +
                                                       (f" — {p.get('calibration_path')}" if p.get('class') == 'NOT_DEFINED_IN_THEORY' else ''))
    for n in SC.REQUIRED_PARAMETERS: L.append(f'| `{n}` | {desc(SC.lookup(strict, n))} | {desc(SC.lookup(exp, n))} |')
    L += ['', '## 4. Verified equivalent optimizations (not model deviations)', '']
    L += [f"- **{o['id']}** ({', '.join(o['equations'])}): {o['what']} — `{o['test']}`" for o in SC.EQUIVALENT_OPTIMIZATIONS]
    L += ['', '## 5a. Audit b62c7cee findings', '', '| finding | status | response |', '|---|---|---|']
    L += [f'| {k} | {v[0]} | {v[1]} |' for k, v in SC.AUDIT_FINDINGS.items()]
    L += ['', '## 5. THEORY_CONFLICT records', '']
    for c in SC.THEORY_CONFLICTS: L += [f"- **{c['id']}** ({c['status']}): theory — {c['theory']} spec — {c['spec']} Consequence — {c['consequence']} Implementation — {c['implementation']}"]
    L += ['', '## 6. Strict-mode stage blocking', '', 'Strict mode (default of `tools/run_svgd_support.py`) stops at the first stage whose parameters are not usable:', '']
    for st, ps in SC.STAGE_PARAMETERS.items():
        L.append(f"- `{st}`: " + (', '.join(f'`{p}`' for p in ps) if ps else 'no statistical parameters'))
    L += ['', 'Two levels of success are never conflated: operator rows above are `THEORY_EXACT_ON_DEFINED_MODEL` evidence; physical motion recovery is reported '
          'separately (`gate4_measurements` in the parity summary, `reports/svgd_support_implementation_and_validation.md`).', '']
    (REPO/'docs/SVGD_THEORY_TO_CODE_CONTRACT.md').write_text('\n'.join(L)+'\n')


if __name__ == '__main__':
    sys.exit(main())
