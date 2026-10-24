#!/usr/bin/env python3
"""Explicit real-data TEST prefixes comparing computational group occupancy."""
import argparse,json,subprocess,sys,time
from pathlib import Path
import numpy as np
REPO=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO))
from rgbd_common.array_store import ArrayStore


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--fps',type=float);parser.add_argument('--frames',type=int,default=8)
    parser.add_argument('--occupancies',type=int,nargs='+',default=[64,128,256]);parser.add_argument('--device',default='cuda',choices=['cpu','cuda'])
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True);reports=[]
    for occupancy in args.occupancies:
        root=args.output/f'occupancy_{occupancy}';cmd=[sys.executable,'-B',str(REPO/'tools/run_free_grouped_rgbd_joint.py'),'--input',str(args.input),'--output',str(root),'--device',args.device,'--target-occupancy',str(occupancy),'--frame-limit',str(args.frames),'--anchor','first','--stop-after','phase1','--no-viewer']
        if args.fps is not None:cmd+=['--fps',str(args.fps)]
        start=time.monotonic()
        with (args.output/f'occupancy_{occupancy}.log').open('w') as log:run=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,cwd=REPO)
        if run.returncode!=3:raise RuntimeError(f'Benchmark failed: {root}; exit={run.returncode} (3 is the expected intentionally incomplete test run)')
        result=json.loads((root/'result.json').read_text());summary=json.loads((root/'first/phase1/summary.json').read_text())
        if result['full_source']:raise AssertionError('Benchmark must be an explicitly limited TEST')
        record={'occupancy':occupancy,'frames':summary['frames'],'points':summary['points'],'groups':summary['groups'],'seconds':time.monotonic()-start,'filter_seconds':summary['runtime_seconds'],'backend':result['backend'],'rss_peak_bytes':result['rss_peak_bytes'],'training_factors':0,'conditional_energy_before':0.,'conditional_energy_after':0.}
        residual=[];camera_residual={};store=ArrayStore(root/'first/phase1/factors.sqlite')
        for path in sorted((root/'first/phase1/filter').glob('*.json')):
            report=json.loads(path.read_text());record['conditional_energy_before']+=float(np.sum(report['before']));record['conditional_energy_after']+=float(np.sum(report['after']))
            row=store.read(str(report['frame']));record['training_factors']+=len(row['z']);residual.extend(row['residual'])
            if [v['camera_index'] for v in report['association']]!=list(range(summary['cameras'])):raise AssertionError('Camera excluded from benchmark')
            for ci in range(summary['cameras']):camera_residual.setdefault(str(ci),[]).extend(row['residual'][row['camera']==ci])
        store.close();e=np.asarray(residual)
        record['uvz_rmse_pixels_pixels_metres']=np.sqrt(np.mean(e*e,axis=0)).tolist() if len(e) else None
        record['camera_policy']='all_available'
        record['per_camera']={ci:{'raw_returns':len(v),'uvz_rmse_pixels_pixels_metres':np.sqrt(np.mean(np.asarray(v)**2,axis=0)).tolist() if len(v) else None} for ci,v in camera_residual.items()}
        reports.append(record);print(json.dumps(record),flush=True)
    (args.output/'comparison.json').write_text(json.dumps({'scope':'Same real input prefix, all cameras; computational group resolution benchmark only, not complete reconstruction or physical group ablation','results':reports},indent=2)+'\n')


if __name__=='__main__':main()
