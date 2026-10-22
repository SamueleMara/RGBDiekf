"""Rebuild an immutable accepted checkpoint score and conditional uncertainty.

This is a checkpoint audit, not a completed reconstruction or convergence test.
The running worker's checkpoint can advance while this fixed snapshot is read.
"""
import argparse,hashlib,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from rgbd_common.backends.torch_cuda import Backend
from rgbd_common.config import load_config
from rgbd_common.dataloader import load_dataset
from rgbd_common.frame_cache import CachedDataset
from rgbd_common.geometry import PointMap
from rgbd_common.lie import State
from rgbd_common.score import score_full_sequence
from rgbd_common.map_uncertainty import conditional_map_uncertainty
from rgbd_common.source_factors import export_source_factors


def main():
    p=argparse.ArgumentParser();p.add_argument('--input',required=True,type=Path);p.add_argument('--run',required=True,type=Path);p.add_argument('--output',required=True,type=Path);p.add_argument('--score-only',action='store_true');a=p.parse_args()
    if a.output.exists():raise FileExistsError('Choose a fresh checkpoint audit directory')
    a.output.mkdir(parents=True);start=time.monotonic();meta=json.loads((a.run/'checkpoint.json').read_text())
    model=PointMap.load(a.run/meta['point_file'])
    with np.load(a.run/meta['trajectory_file']) as trace:states=[State(p,v) for p,v in zip(trace['poses'],trace['twists'])]
    source=hashlib.sha256()
    repo=Path(__file__).resolve().parents[1]
    for file in sorted(repo.glob('rgbd_common/**/*.py')):source.update(str(file.relative_to(repo)).encode());source.update(file.read_bytes())
    backend=Backend('cuda');config=load_config();dataset=CachedDataset(load_dataset(a.input),a.run/'raw_cache')
    score=score_full_sequence(model,states,dataset,config,backend,description='Rebuild immutable checkpoint score')
    compatible=bool(np.isclose(score['energy'],meta['energy'],rtol=1e-12,atol=1e-5))
    report={'purpose':'Accepted checkpoint audit; not full-cycle or convergence acceptance','source_common_sha256':source.hexdigest(),
            'checkpoint':meta,'frames':len(states),'cameras':len(dataset.cameras),'recomputed_score':score,
            'score_difference':score['energy']-meta['energy'],'passes_unchanged_resume_guard':compatible}
    if not a.score_only:
        model.save(a.output/'audited_points.npz');np.savez(a.output/'audited_trajectory.npz',poses=np.array([s.poses for s in states]),twists=np.array([s.twists for s in states]))
        report['map_uncertainty']=conditional_map_uncertainty(model,states,dataset,config,backend,a.output/'map_uncertainty')
        report['source_coupling']=export_source_factors(model,states,dataset,config,backend,a.output/'source_pose_factors')
    report.update(gpu=backend.memory(),runtime_seconds=time.monotonic()-start)
    (a.output/'checkpoint_audit.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({key:report[key] for key in ('frames','cameras','score_difference','passes_unchanged_resume_guard','runtime_seconds','gpu')},indent=2),flush=True)
    return 0 if compatible else 1


if __name__=='__main__':raise SystemExit(main())
