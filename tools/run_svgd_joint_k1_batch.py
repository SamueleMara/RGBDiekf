#!/usr/bin/env python3
"""Sequential batch of joint K = 1 runs + their separate-process post-hoc oracles (GT only after each run has finished).

    python -B tools/run_svgd_joint_k1_batch.py --plan overnight --output-root ROOT/svgd_joint_k1_batch_<stamp>
    python -B tools/run_svgd_joint_k1_batch.py --plan overnight --output-root ROOT/svgd_joint_k1_batch_<stamp> --resume   # continue a stopped batch

--resume keeps every finished job (run DONE and oracle DONE or not needed), re-runs only a missing oracle of a finished run, and re-runs an
interrupted job into a NEW folder <name>_r<k> (the interrupted folder is kept untouched). The oracle is a separate process with a wall-clock
limit (--oracle-timeout); hitting it is recorded as oracle TIMEOUT and the batch continues.

Each job: ROOT/<name>/ (run) + ROOT/<name>.log; the oracle writes ROOT/<name>/posthoc/k1_oracle/. ROOT/batch_status.json is rewritten
after every step (progress for tools/run_status.py). A failed job is recorded and the batch continues.
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PY = sys.executable
SYN = '/media/samuelemara/Extreme SSD/Samuele/Backup_09_10_26/Sapien/output/preparation_8cams/partnet_subset_fixed_render_linear'
REAL = '/media/samuelemara/Extreme SSD/Samuele/Backup_09_10_26/Sapien/output/real'


def plan(name, cfg):
    if name != 'overnight': raise SystemExit(f'unknown plan {name}')
    jobs = [dict(name=f'synthetic_{m}', synthetic=['--motion', mo]+extra+['--frames', '10']) for m, mo, extra in
            (('static', 'none', []), ('prismatic_textured', 'translate', []), ('prismatic_uniform_colour', 'translate', ['--uniform-colour']), ('hinge_0p9rad', 'rotate', []))]
    jobs += [dict(name='USB_100109_10f', scene=f'{SYN}/USB_100109', frames=10, gt=True), dict(name='storage_45135_4f', scene=f'{SYN}/storage_45135', frames=4, gt=True)]
    jobs += [dict(name=f'red_fridge_{i:06d}_2f', scene=f'{REAL}/final_red_fridge_episode_{i:06d}', frames=2, gt=False) for i in range(14)]
    for j in jobs: j['config'] = str(cfg)
    return jobs


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--plan', default='overnight'); ap.add_argument('--output-root', type=Path, required=True)
    ap.add_argument('--config', type=Path, default=REPO/'configs/svgd_joint_k1_overnight.json'); ap.add_argument('--device', default='cuda')
    ap.add_argument('--resume', action='store_true'); ap.add_argument('--oracle-timeout', type=float, default=3600., help='seconds per oracle process')
    a = ap.parse_args(); root = a.output_root; jobs = plan(a.plan, a.config); old = {}
    if a.resume:
        st = json.loads((root/'batch_status.json').read_text())
        if st['plan'] != a.plan or st['config'] != str(a.config): raise SystemExit('--resume: plan/config differ from the stopped batch')
        old = {r['name']: r for r in st['jobs']}; st['jobs'] = []; st.pop('finished', None); st.setdefault('resumed', []).append(time.strftime('%Y-%m-%d %H:%M:%S'))
    else:
        if root.exists() and any(root.iterdir()): raise SystemExit(f'{root} exists and is not empty (use --resume to continue it)')
        root.mkdir(parents=True); st = {'plan': a.plan, 'config': str(a.config), 'jobs': [], 'started': time.strftime('%Y-%m-%d %H:%M:%S')}
    save = lambda: (root/'batch_status.json').write_text(json.dumps(st, indent=1))
    for i, j in enumerate(jobs):
        prev = old.get(j['name']); needs_oracle = bool('synthetic' in j or j.get('gt'))
        if prev and prev['state'] == 'DONE' and (prev.get('oracle') == 'DONE' or not needs_oracle):
            try: prev['oracle_status'] = json.loads(sorted((Path(prev['output'])/'posthoc').glob('k1_oracle*/oracle.json'), key=lambda q: q.stat().st_mtime)[-1].read_text())['status']
            except Exception: pass                                                      # latest post-hoc verdict (e.g. a manual re-check)
            st['jobs'].append(prev); save(); continue                                   # finished: kept as is
        name = j['name']
        if prev and prev['state'] != 'DONE':                                            # interrupted run: never overwrite its folder
            k = 1
            while (root/f'{name}_r{k}').exists(): k += 1
            name = f'{name}_r{k}'
        out = root/name; log = root/f"{name}.log"; t0 = time.time()
        rec = {'name': name, 'index': i+1, 'of': len(jobs), 'state': 'RUNNING', 'output': str(out)}
        st['jobs'].append(rec); st['current'] = name; save()
        if 'synthetic' in j:
            cmd = [PY, '-B', str(REPO/'tools/run_svgd_joint_k1_synthetic.py'), *j['synthetic'], '--config', j['config'], '--device', a.device, '--output', str(out)]
            ocmd = [PY, '-B', str(REPO/'tools/audit_svgd_joint_k1_oracles.py'), '--run', str(out), '--synthetic', '--device', a.device]
        else:
            cmd = [PY, '-B', str(REPO/'tools/run_svgd_joint_k1.py'), '--scene', j['scene'], '--config', j['config'], '--num-moving-parts', '1', '--device', a.device,
                   '--frame-limit', str(j['frames']), '--output', str(out), '--debug', 'none']
            ocmd = [PY, '-B', str(REPO/'tools/audit_svgd_joint_k1_oracles.py'), '--run', str(out), '--scene', j['scene'], '--device', a.device] if j.get('gt') else None
        if prev and prev['state'] == 'DONE':                                            # run finished, only its oracle is missing
            rec.update(state='DONE', run_seconds=prev.get('run_seconds')); rc = 0
        else:
            with open(log, 'w') as f: rc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=REPO).returncode
            rec.update(state='DONE' if rc == 0 else f'FAILED (exit {rc})', run_seconds=time.time()-t0); save()
        if rc == 0 and ocmd:
            olog = root/f"{name}.oracle.log"; olog = olog if not olog.exists() else root/f"{name}.oracle_{int(time.time())}.log"
            rec['oracle'] = 'RUNNING'; rec['oracle_started'] = time.time(); save()
            try:
                with open(olog, 'w') as f: orc = subprocess.run(ocmd, stdout=f, stderr=subprocess.STDOUT, cwd=REPO, timeout=a.oracle_timeout).returncode
                rec['oracle'] = 'DONE' if orc == 0 else f'FAILED (exit {orc})'
            except subprocess.TimeoutExpired: rec['oracle'] = f'TIMEOUT ({a.oracle_timeout:.0f} s)'
            try: rec['oracle_status'] = json.loads(sorted((out/'posthoc').glob('k1_oracle*/oracle.json'), key=lambda q: q.stat().st_mtime)[-1].read_text())['status']
            except Exception: pass
        try: rec['run_status'] = json.loads((out/'svgd_joint_k1/status.json').read_text())
        except Exception: pass
        rec['seconds'] = time.time()-t0; save()
    st['finished'] = time.strftime('%Y-%m-%d %H:%M:%S'); st['current'] = None; save()


if __name__ == '__main__':
    main()
