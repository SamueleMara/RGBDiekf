#!/usr/bin/env python3
"""Compose reports/rigeuler_rgbd_theory_validation.md from the machine-readable artifacts.

Inputs: reports/rigeuler_rgbd_test_matrix.csv, reports/image_iekf_dense_covariance_comparison.json,
reports/coherence_occlusion_simulation.json, reports/ablations.json, and for every --run a pipeline output folder with its
result.json and (optional) gt_evaluation.json written by tools/evaluate_image_iekf_gt.py after inference.
The narrative sections (theory, adaptations, failures) are in reports/rigeuler_rgbd_theory_validation_text.md and are
included verbatim; this script only renders numbers.
"""
import argparse, csv, json
from pathlib import Path
REPO = Path(__file__).resolve().parents[1]
R = REPO/'reports'


def load(path):
    p = Path(path)
    return json.loads(p.read_text()) if p.exists() else None


def table(rows, header):
    out = ['| '+' | '.join(header)+' |', '|'+'---|'*len(header)]
    out += ['| '+' | '.join('' if v is None else (f'{v:.4g}' if isinstance(v, float) else str(v)) for v in r)+' |' for r in rows]
    return '\n'.join(out)


def gates():
    p = R/'rigeuler_rgbd_test_matrix.csv'
    if not p.exists(): return '_test matrix not generated_', {}
    rows = list(csv.DictReader(open(p)))
    by = {}
    for r in rows: by.setdefault(r['contract_section'], []).append(r)
    summary = [[sec, sum(r['outcome'] == 'PASS' for r in rs), sum(r['outcome'] in ('FAIL', 'ERROR') for r in rs), len(rs)] for sec, rs in by.items()]
    failed = [[r['test'].split('.')[-1], r['outcome'], r['detail']] for r in rows if r['outcome'] in ('FAIL', 'ERROR')]
    text = table(summary, ['contract section', 'pass', 'fail/error', 'total'])
    if failed: text += '\n\nFailing gates:\n\n'+table(failed, ['test', 'outcome', 'detail'])
    return text, {sec: all(r['outcome'] == 'PASS' for r in rs) for sec, rs in by.items()}


def covariance():
    d = load(R/'image_iekf_dense_covariance_comparison.json')
    if not d: return '_not generated_'
    rows = []
    for f in d['fixtures']:
        last = f['frames'][-1]
        rows.append([f"{f['points']} pts / {f['groups']} groups / {f['direction']}", max(r['structured_vs_actual_rel'] for r in f['frames']),
                     last['actual_over_optimal_position_trace'], last['assumed_over_actual_position_trace'], last['rigeuler_over_actual_position_trace']])
    return table(rows, ['fixture (5 updates)', 'max rel. error structured vs dense', 'actual/optimal pos. trace', 'assumed-density/actual pos. trace', 'RigEuler/actual pos. trace'])


def coherence():
    d = load(R/'coherence_occlusion_simulation.json')
    if not d: return '_not generated_'
    return d['metric']+'\n\n'+table([[k, v['reference'], v['zoh'], v['supported']] for k, v in d['scenarios'].items()], ['scenario', 'pinned reference', 'zero-order hold', 'supported (production)'])


def ablations():
    d = load(R/'ablations.json')
    if not d: return '_not generated_'
    rows = []
    for motion, res in d.items():
        if motion == 'covariance': continue
        for k, v in res.items():
            rows.append([motion, k, json.dumps(v)])
    return table(rows, ['scene', 'variant', 'result (GT used for evaluation only)'])


def run_section(run):
    run = Path(run)
    res, gt = load(run/'result.json'), load(run/'gt_evaluation.json')
    lines = [f'### `{run}`', '']
    if res: lines += ['```json', json.dumps({k: res[k] for k in ('status', 'directions', 'runtime_seconds') if k in res}, indent=1), '```', '']
    for anchor in ('first', 'last'):
        j = load(run/anchor/'joint/result.json')
        if j:
            lines.append(f"**{anchor}** joint: `{j['status']}` - {j['reason']}")
            lines.append(table([[k, v.get('n'), v.get('mean_differential'), v.get('statistic'), v.get('p_value')] for k, v in j['tests'].items()],
                               ['comparison', 'frames', 'mean NLL differential', 'HLN statistic', 'p']))
            lines.append('')
        s = load(run/anchor/'support/result.json')
        if s: lines += [f"**{anchor}** support counts: `{json.dumps(s['counts'])}`, MAP converged {s['map']['converged']}", '']
    if gt: lines += ['Ground-truth evaluation (after inference):', '```json', json.dumps(gt, indent=1)[:6000], '```', '']
    return '\n'.join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__); ap.add_argument('--run', nargs='*', default=[]); a = ap.parse_args()
    text = (R/'rigeuler_rgbd_theory_validation_text.md').read_text() if (R/'rigeuler_rgbd_theory_validation_text.md').exists() else ''
    g, ok = gates()
    body = [text, '## Measured gates (tools/test_matrix.py)', '', g, '',
            '## Covariance: structured exact vs dense oracles, assumed density, RigEuler', '', covariance(), '',
            '## Coherence occlusion simulations', '', coherence(), '', '## Ablations (tools/run_ablations.py)', '', ablations(), '',
            '## Complete runs', '']+[run_section(r) for r in a.run]
    (R/'rigeuler_rgbd_theory_validation.md').write_text('\n'.join(body))
    print('written', R/'rigeuler_rgbd_theory_validation.md')


if __name__ == '__main__':
    main()
