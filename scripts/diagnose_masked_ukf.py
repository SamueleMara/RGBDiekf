"""Preserve the failing four-frame fixture; never a real reconstruction claim."""
import argparse,json,sys
from pathlib import Path
import numpy as np


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    repo=Path(__file__).resolve().parents[1];sys.path[:0]=[str(repo),str(repo/'tests')]
    from parallel_fixtures import DrawerScene,configuration
    from rgbd_common.backends.torch_cuda import Backend
    from rgbd_common.outer_loop import run_outer
    from rgbd_common.outputs import export_result
    from ukf.filter import UKF
    if args.output.exists():raise FileExistsError('Preserve existing diagnostic; use a fresh output path')
    args.output.mkdir(parents=True);scene=DrawerScene();config=configuration();config.numerical['inner_iterations']=4;backend=Backend('cpu')
    result=run_outer(UKF(config,backend),scene,config,backend,args.output,max_outer=1,visual=False)
    rows=[]
    for path in sorted(args.output.glob('outer_000/filter/step_*.json')):
        data=json.loads(path.read_text());d=data['diagnostics'];groups=d.get('group_diagnostics',[d])
        rows.append({'frame':d['frame'],'matched_raw_factors':d['matched_factors'],'groups':[{'active':g.get('active_factors'),'deferred':g.get('deferred_visibility_factors'),'proposal_inf':g.get('proposal_inf')} for g in groups]})
    measured=float(max(np.linalg.norm(s.poses[:,:3,3],axis=1).max() for s in result.states))
    report=export_result(result,scene,config,backend,args.output,'UKF')
    diagnostic={'fixture':'synthetic four-frame translating occluder; not the real scene','known_fixture_translation_per_frame_m':.04,'accepted_max_translation_m':measured,'required_motion_assertion_pass':measured>.05,'filter_factor_counts':rows,'pipeline_report':report,'qualification':'Full-map sigma visibility is evaluated without the old fixed-bin approximation. Zero active factors retain the predicted prior, never invent measurement information. Does not satisfy physical motion-recovery acceptance.'}
    (args.output/'motion_recovery_diagnostic.json').write_text(json.dumps(diagnostic,indent=2)+'\n');print(json.dumps({key:diagnostic[key] for key in ('fixture','accepted_max_translation_m','required_motion_assertion_pass','filter_factor_counts')},indent=2))
    return int(not diagnostic['required_motion_assertion_pass'])


if __name__=='__main__':raise SystemExit(main())
