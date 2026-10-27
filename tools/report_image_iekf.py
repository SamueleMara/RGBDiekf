#!/usr/bin/env python3
"""Write the measured validation report of one image-IEKF run (numbers read from its outputs only)."""
import argparse, json
from collections import Counter
from pathlib import Path
import numpy as np


def load(p):
    p = Path(p)
    return json.loads(p.read_text()) if p.exists() else None


def frame_table(folder, K):
    rows = []
    for k in range(K):
        r = load(folder/f'{k:06d}.json')
        if not r or 'cameras' not in r or 'stop' not in r: continue      # anchor frame: no update
        cams = r['cameras']
        rows.append({'frame': k, 'stop': r.get('stop'), 'accepted': r.get('accepted_steps'), 'seconds': r.get('seconds'),
                     'rgb_rmse': np.mean([c['rgb_rmse'] for c in cams if c['rgb_rmse'] is not None]),
                     'depth_rmse_m': np.mean([c['depth_rmse_m'] for c in cams if c['depth_rmse_m'] is not None]),
                     'coverage': sum(c['covered_domain_pixels'] for c in cams)/max(1, sum(c['domain_pixels'] for c in cams)),
                     'contributors': sum(c.get('active_contributors', 0) for c in cams),
                     'adjoint': (r.get('jvp_vjp_adjoint_check') or {}).get('relative_error')})
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run', type=Path, required=True); ap.add_argument('--gt', type=Path); ap.add_argument('--tests', type=Path)
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args(); R = a.run
    res, man = load(R/'result.json'), load(R/'manifest.json')
    K = len(man['dataset']['timestamps_seconds'])
    L = ['# Rendered-image RGB-D IEKF — measured validation', '',
         f"Run: `{R}`. Status: **{res['status']}**. Runtime {res['runtime_seconds']/60:.1f} min. Peak RSS {res['rss_peak_bytes']/1e9:.2f} GB. "
         f"CUDA peak allocated {res['backend'].get('peak_allocated_bytes', 0)/1e9:.2f} GB on {res['backend'].get('name', res['backend']['device'])}.",
         f"Measurement model `{man['measurement_model']}`; association used: {man['association_used']}; cameras {man['production_camera_indices']}; frames {K}; full source {res['full_source']}.", '']
    if a.tests and a.tests.exists(): L += ['## Executed tests', '', a.tests.read_text().strip(), '']
    for anchor in ('first', 'last'):
        d = R/anchor
        if not d.exists(): continue
        L += [f'## Direction `{anchor}`', '']
        for stage, folder in (('Phase 1 (free points)', d/'phase1/frames'), ('Phase 2 (rigid)', d/'rigid/frames')):
            rows = frame_table(folder, K)
            if not rows: continue
            sec = [r['seconds'] for r in rows if r['seconds']]
            L += [f'### {stage}', '', f'{len(rows)} updated frames; mean {np.mean(sec):.1f} s/frame; stops {dict(Counter(r["stop"] for r in rows))}.', '',
                  '| frame | stop | accepted | RGB RMSE | depth RMSE (m) | covered domain | active contributors | adjoint rel. err |', '|---|---|---|---|---|---|---|---|']
            for r in rows[::max(1, len(rows)//12)]+[rows[-1]]:
                adjoint = '—' if r['adjoint'] is None else f"{r['adjoint']:.1e}"
                L.append(f"| {r['frame']} | {r['stop']} | {r['accepted']} | {r['rgb_rmse']:.4f} | {r['depth_rmse_m']:.4f} | {r['coverage']:.3f} | {r['contributors']} | {adjoint} |")
            L.append('')
        coh = load(d/'coherence/result.json')
        if coh: L += ['### Coherence', '', f"{coh['status']}; {coh.get('core_nodes')} of {coh.get('representative_nodes')} representative nodes; {coh.get('core_points')} core points; weighting: {coh.get('member_weighting')}.",
                      f"Image evidence: {json.dumps(coh.get('image_evidence'))}", '']
        rig = load(d/'rigid/result.json')
        if rig: L += ['### Rigid SE(3) IEKF', '', f"{rig['status']}; {rig['core_points']} core points; {json.dumps(rig['trajectory_diagnostics'])}", '']
        j = load(d/'joint/result.json')
        if j:
            L += ['### Joint', '', f"**{j['status']}** — {j['reason']}", f"gamma_1dof {j['gamma_1dof']:.4f}; image NLL {json.dumps({k: round(v, 1) for k, v in j['image_nll'].items()})}; "
                  f"max core displacement {j.get('max_core_displacement_m', float('nan')):.3f} m; revolute axis/part scale {j['revolute_axis_over_part_scale']:.3g}.", '']
            for t, c in j['candidates'].items():
                L.append(f"* {t}: axis {np.round(c['axis'], 4).tolist()}" + (f", axis point {np.round(c['axis_point'], 3).tolist()}" if 'axis_point' in c else '') +
                         f", image NLL {c['image_nll']:.1f}, free-rigid vs joint point discrepancy (image q) rms {c['point_discrepancy_image_q_m']['rms']:.4f} m")
            L.append('')
        s = load(d/'support/result.json')
        if s: L += ['### Support completion', '', json.dumps(s['counts']), json.dumps(s['evidence']), '']
    b = load(R/'bidirectional/result.json')
    if b: L += ['## Forward/backward', '', f"**{b['status']}** — {b['reason']}", '```json', json.dumps(b['candidate_comparison'], indent=1), '```', '']
    qa = sorted((R/'viewer').glob('qa_*.json'))
    if qa:
        L += ['## Browser QA (headless Chrome, real WebGL page)', '', '| panel | checks | pass |', '|---|---|---|']
        for q in qa:
            v = load(q); L.append(f"| {q.stem} | {json.dumps(v['checks'])} | {v['pass']} |")
        L.append('')
    if a.gt and a.gt.exists():
        L += ['## Ground-truth evaluation (after inference; never read by the estimator)', '', '```json', a.gt.read_text().strip(), '```', '']
    a.out.write_text('\n'.join(L)+'\n')
    print(a.out)


if __name__ == '__main__':
    raise SystemExit(main())
