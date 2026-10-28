#!/usr/bin/env python3
"""Progress and time left of a running (or finished) train.py run: a dataset-root batch folder or one scene run.

    python tools/run_status.py <output folder>              # once
    python tools/run_status.py <output folder> --watch 10   # refresh every 10 s (Ctrl-C to stop)

Reads only what the run writes: the per-scene logs (<batch>/<scene>.log, or <run>.log next to a single run), the
'[forward] i/n' frame lines of the free and rigid passes, the stage markers, run.lock (held while a scene runs) and
the result.json / summary.json of finished scenes. Stage weights come from the stage timings of the scenes already
finished in the same batch (defaults from a storage run otherwise), so the estimate sharpens as the batch advances.
"""
import argparse, fcntl, json, os, re, sys, time
from pathlib import Path

PHASES = ['free_forward', 'free_backward', 'coherence_forward', 'coherence_backward', 'rigid_forward', 'rigid_backward',
          'joint_forward', 'joint_backward', 'support_forward', 'support_backward', 'debug', 'evaluate_qa']
DEFAULT_W = {'free_forward': 86, 'free_backward': 91, 'coherence_forward': 14, 'coherence_backward': 14, 'rigid_forward': 15, 'rigid_backward': 16,
             'joint_forward': 3.5, 'joint_backward': 2, 'support_forward': 0, 'support_backward': 0, 'debug': 18, 'evaluate_qa': 37}   # storage, s
FRAME = re.compile(r'^\[(forward|backward)\] (\d+)/(\d+) .*?(observed|rows)')


def is_scene(p):
    return (Path(p)/'rgb').is_dir() and (Path(p)/'metadata').is_dir()


def running(run):
    """True while train.py holds run.lock (removed at the end; a stale file after a kill is not held)."""
    lock = Path(run)/'run.lock'
    if not lock.exists(): return False
    try:
        with open(lock) as f:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB); fcntl.flock(f, fcntl.LOCK_UN)
        return False
    except OSError:
        return True


def weights(finished):
    """Mean seconds per phase over finished scenes (evaluation + QA = runtime minus the timed stages)."""
    rows = []
    for r in finished:
        st = dict(r.get('stage_seconds') or {}); tot = r.get('runtime_seconds')
        if not st or not tot: continue
        st['evaluate_qa'] = max(tot-sum(v for k, v in st.items() if k in PHASES), 0.)
        rows.append(st)
    if not rows: return dict(DEFAULT_W), False
    return {p: sum(r.get(p, 0.) for r in rows)/len(rows) for p in PHASES}, True


def scene_progress(log, w):
    """(fraction done, phase text) from a scene log."""
    done, cur, frac_cur = set(), None, 0.
    text = Path(log).read_text(errors='replace').splitlines() if Path(log).exists() else []
    kind = None
    for line in text:
        m = FRAME.match(line)
        if m:
            d, i, n, what = m.group(1), int(m.group(2)), int(m.group(3)), m.group(4)
            ph = f"{'free' if what == 'observed' else 'rigid'}_{d}"
            if ph != cur and cur: done.add(cur)
            if what == 'rows': done.update(p for p in PHASES[:4])        # rigid starts after free + coherence
            if ph == 'free_backward': done.add('free_forward')
            if ph == 'rigid_backward': done.add('rigid_forward')
            cur, frac_cur = ph, i/max(n, 1); kind = f'{ph.replace("_", " ")} frame {i}/{n}'
            continue
        for stage in ('coherence', 'rigid', 'joint', 'support'):
            m2 = re.match(rf'^\[{stage} (forward|backward)\]', line)
            if m2:
                ph = f'{stage}_{m2.group(1)}'; done.add(ph)
                done.update(PHASES[:PHASES.index(ph)])
                cur, frac_cur, kind = None, 0., f'{ph.replace("_", " ")} done'
        if line.startswith('Debug index:'): done.update(PHASES[:PHASES.index('debug')+1]); cur, kind = None, 'debug pages written; GT evaluation'
        if line.startswith('evaluate: exit'): kind = 'page QA (headless Chrome)'
    if any(l.strip().startswith('"runtime_s"') for l in text[-3:]): return 1., 'finished'
    tot = sum(w.values()) or 1.
    f = sum(w[p] for p in done)/tot + (w[cur]*frac_cur/tot if cur and cur not in done else 0.)
    return min(f, .999), kind or 'starting (loading data)'


def fmt(s):
    s = max(int(s), 0); h, r = divmod(s, 3600); m, s = divmod(r, 60)
    return f'{h}h{m:02d}m' if h else f'{m}m{s:02d}s'


def status(out, log=None):
    out = Path(out).resolve(); now = time.time(); lines = []
    single = (out/'run_manifest.json').exists() and not (out/'batch.json').exists() and not (out/'summary.json').exists()
    if single:
        scenes = [(out.name, out, Path(log) if log else out.with_name(out.name+'.log'))]
    else:
        names = []
        if (out/'batch.json').exists(): names = json.loads((out/'batch.json').read_text())['scenes']
        else:                                                              # older batches: the dataset root of a finished scene
            for p in sorted(out.iterdir()):
                if (p/'run_manifest.json').exists():
                    root = Path(json.loads((p/'run_manifest.json').read_text())['arguments']['scene']).parent
                    names = [q.name for q in sorted(root.iterdir()) if q.is_dir() and is_scene(q)]; break
            names = names or sorted(p.stem for p in out.glob('*.log') if p.stem != 'batch')
        scenes = [(n, out/n, out/f'{n}.log') for n in names]
    finished, rows = [], []
    for name, run, lg in scenes:
        res = json.loads((run/'result.json').read_text()) if (run/'result.json').exists() else None
        if res and 'runtime_seconds' in res and not running(run): finished.append(res)
    w, calibrated = weights(finished)
    durations = [r['runtime_seconds'] for r in finished]
    mean_d = sum(durations)/len(durations) if durations else sum(w.values())
    left_total, n_done, n_run, n_wait = 0., 0, 0, 0
    for name, run, lg in scenes:
        res = json.loads((run/'result.json').read_text()) if (run/'result.json').exists() else None
        if run.exists() and running(run):
            n_run += 1
            start = (run/'run.lock').stat().st_mtime; el = now-start
            f, what = scene_progress(lg, w)
            est = el/f if f >= .08 else max(mean_d, el)
            left = max(est-el, 0.); left_total += left
            rows.append(f'  RUNNING  {name:<20} {what:<34} {100*f:5.1f}%  elapsed {fmt(el):>7}  left ~{fmt(left)}')
        elif res and 'runtime_seconds' in res:
            n_done += 1
            rows.append(f'  done     {name:<20} {res.get("status", "?"):<34} {fmt(res["runtime_seconds"]):>7}')
        elif run.exists() and any(run.iterdir()):
            n_done += 1
            rows.append(f'  STOPPED  {name:<20} no result.json and no live process (failed or killed): see {lg.name}')
        else:
            n_wait += 1; left_total += mean_d
            rows.append(f'  queued   {name:<20}')
    head = f'{out}\n{n_done}/{len(scenes)} done, {n_run} running, {n_wait} queued'
    if n_run or n_wait:
        head += f'   |   time left ~{fmt(left_total)}  (finish ~{time.strftime("%H:%M", time.localtime(now+left_total))})'
        head += f'   [stage weights {"from this batch" if calibrated else "default (storage)"}; queued scene ~{fmt(mean_d)}]'
    else:
        head += '   |   finished'
    if (out/'summary.csv').exists(): rows.append(f'  summary: {out/"summary.csv"}')
    return head+'\n'+'\n'.join(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('output', type=Path, help='batch output folder (dataset root run) or single scene run folder')
    ap.add_argument('--log', type=Path, help='stdout log of a single-scene run (default: <run>.log next to the run folder)')
    ap.add_argument('--watch', type=float, default=0, help='refresh every N seconds')
    a = ap.parse_args()
    if not a.watch:
        print(status(a.output, a.log)); return 0
    try:
        while True:
            s = status(a.output, a.log)
            sys.stdout.write('\033[2J\033[H'+time.strftime('%H:%M:%S')+'  '+s+'\n'); sys.stdout.flush()
            if 'finished' in s.splitlines()[1]: return 0
            time.sleep(a.watch)
    except KeyboardInterrupt:
        return 0


if __name__ == '__main__':
    raise SystemExit(main())
