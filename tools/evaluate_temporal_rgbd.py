#!/usr/bin/env python3
"""Independent post-run temporal diagnostics; GT is opt-in in another process."""
import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from utils.io import json_write,serial


def evaluate(run):
    out={'run':str(run),'ground_truth_used':False,'directions':{}}
    for direction in ('forward','backward'):
        root=run/f'free_{direction}'
        if not (root/'status.json').exists():continue
        status=json.loads((root/'status.json').read_text());rec={'status':status}
        files=sorted((root/'temporal').glob('frame_*.json'))
        frames=[json.loads(p.read_text()) for p in files]
        rec['frames']=frames
        if frames:
            rec['summary']={'temporal_rows':sum(f['rows'] for f in frames),'prediction_only_fraction':sum(f['prediction_only'] for f in frames)/sum(f['observed_points']+f['prediction_only'] for f in frames),
                            'outside_foreground_point_camera_events':sum(f['rendered_outside_foreground'] for f in frames),
                            'unexplained_foreground_samples':sum(f['unexplained_foreground'] for f in frames),
                            'owner_switches':sum(f['switched_pixels'] for f in frames),'seconds':sum(f['seconds'] for f in frames),
                            'convergence_fraction':sum(f['solver']['converged'] for f in frames)/len(frames)}
        elif not (root/'temporal').exists():
            legacy=[json.loads(p.read_text()) for p in sorted((root/'numeric').glob('frame_*.json'))]
            if legacy:
                total=sum(n['sensor']['unique_points_observed']+n['sensor']['prediction_only_points'] for n in legacy)
                rec['summary']={'tracking_rows':sum(n['sensor']['rows_accepted_final'] for n in legacy),
                    'prediction_only_fraction':sum(n['sensor']['prediction_only_points'] for n in legacy)/total,
                    'seconds':sum(n['processing']['wall_ms'] for n in legacy)/1000,
                    'observation_model':'legacy; different likelihood, not a temporal objective comparison'}
        if (root/'temporal/innovation_summary.csv').exists():
            with (root/'temporal/innovation_summary.csv').open() as f:rec['innovations']=list(csv.DictReader(f))
        if (root/'velocity.npy').exists():
            V=np.load(root/'velocity.npy',mmap_mode='r');k=1 if direction=='forward' else len(V)-2
            rec['first_step_speed_m_s_quantiles']=np.quantile(np.linalg.norm(np.asarray(V[k],float),axis=1),[0,.5,.95,.99,1]).tolist()
        rec['static_error_or_NEES']='NOT_COMPUTED_WITHOUT_POSTHOC_GT'
        out['directions'][direction]=rec
    if not out['directions']:
        failure=run/'failure.json';out['failure']=json.loads(failure.read_text()) if failure.exists() else 'NO_COMPLETED_PASS'
    return out


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--run',required=True,type=Path);ap.add_argument('--scene',type=Path)
    ap.add_argument('--with-gt',action='store_true');ap.add_argument('--output',type=Path)
    a=ap.parse_args();dest=a.output or a.run/'evaluation_temporal';dest.mkdir(parents=True,exist_ok=True)
    report=evaluate(a.run);json_write(dest/'diagnostics.json',serial(report))
    text='# Temporal RGB-D diagnostics\n\nGT-free diagnostics; no quantities are passed back into inference.\n\n'
    for d,r in report['directions'].items():text+=f'## {d}\n\n```json\n{json.dumps(r.get("summary",r["status"]),indent=2)}\n```\n\n'
    (dest/'report.md').write_text(text)
    if a.with_gt:
        if a.scene is None:ap.error('--with-gt requires --scene')
        return subprocess.call([sys.executable,str(Path(__file__).with_name('evaluate_projective_gt.py')),'--run',str(a.run),'--scene',str(a.scene),'--out',str(dest/'ground_truth')])
    print(dest/'report.md');return 0

if __name__=='__main__':raise SystemExit(main())
