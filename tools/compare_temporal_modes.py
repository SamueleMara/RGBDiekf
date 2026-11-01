#!/usr/bin/env python3
"""Compare actual run diagnostics; failures and absent experiments remain explicit."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tools.evaluate_temporal_rgbd import evaluate
from utils.io import json_write,serial


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--root',type=Path,required=True);a=ap.parse_args()
    results=[]
    for p in sorted(a.root.rglob('run_manifest.json')):
        if p.parent.name.startswith(('free_','rigid_')):continue
        man=json.loads(p.read_text());record=evaluate(p.parent)
        record['mode']=man.get('temporal',{}).get('mode',man.get('observation_model','legacy'))
        record['scene']=str(man.get('arguments',{}).get('scene','unknown'));results.append(record)
    for p in sorted(a.root.rglob('failure.json')):
        if not (p.parent/'run_manifest.json').exists():results.append({'run':str(p.parent),'failure':json.loads(p.read_text()),'mode':'preflight failure'})
    json_write(a.root/'temporal_comparison.json',serial(results))
    lines=['# Temporal modes: measured runs only','', '| Run | Mode | Direction | Status | No-row fraction | Rows | Seconds |','|---|---|---|---|---:|---:|---:|']
    for r in results:
        if not r.get('directions'):lines.append(f'| {r["run"]} | {r["mode"]} | — | FAILED / NOT RUN | — | — | — |')
        for d,rec in r.get('directions',{}).items():
            s=rec.get('summary',{});lines.append(f'| {Path(r["run"]).name} | {r["mode"]} | {d} | {rec["status"]["status"]} | {s.get("prediction_only_fraction","—")} | {s.get("temporal_rows",s.get("tracking_rows","—"))} | {s.get("seconds","—")} |')
    lines+=['','A completed pass is not a scientific validation. Static/moving accuracy and NEES require separate post-hoc GT evaluation. Missing experiments are not imputed.']
    (a.root/'temporal_comparison.md').write_text('\n'.join(lines)+'\n');print(a.root/'temporal_comparison.md')

if __name__=='__main__':main()
