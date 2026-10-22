"""Known-motion synthetic accuracy diagnostic, separate from real acceptance."""
import argparse,json,sys
from pathlib import Path
sys.path[:0]=[str(Path(__file__).resolve().parents[1]),str(Path(__file__).resolve().parents[1]/'tests')]
import numpy as np
from parallel_fixtures import DrawerScene,configuration
from rgbd_common.backends.torch_cuda import Backend
from rgbd_common.outer_loop import run_outer
from rgbd_common.outputs import export_result
from rgbd_common.lie import exp,inverse,log
from iekf.filter import IEKF
from ukf.filter import UKF


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True,type=Path);args=p.parse_args()
    if args.output.exists():raise FileExistsError('Preserving existing diagnostic; select a fresh output directory')
    args.output.mkdir(parents=True);motion=np.array([.04,.002,-.001,.005,-.008,.01]);scene=DrawerScene(motion_per_frame=motion)
    true=exp((len(scene.times)-1)*motion);reports=[]
    for cls in (IEKF,UKF):
        directory=args.output/cls.name.lower();config=configuration();config.numerical['inner_iterations']=4
        result=run_outer(cls(config,Backend('cpu')),scene,config,Backend('cpu'),directory,max_outer=1,visual=False)
        report=export_result(result,scene,config,Backend('cpu'),directory,cls.name)
        errors=[]
        for g,t in enumerate(result.states[-1].poses):
            # Ground truth used ONLY here, after inference has completed.
            error=log(t@inverse(true))
            errors.append({'group':g,'points':int((result.model.group==g).sum()),'translation_error_m':float(np.linalg.norm(t[:3,3]-true[:3,3])),
                           'rotation_error_radians':float(np.linalg.norm(error[3:])),'estimated_final_pose':t.tolist()})
        reports.append({'pipeline':cls.name,'status':result.status,'initial_points':result.initial_points,'final_points':result.model.count,
                        'groups':len(errors),'final_energy':report['final_energy'],'group_errors_against_moving_body':errors,
                        'qualification':'Fixture also contains a stationary rear body. No claim every group should follow the moving body. Extra moving-body splits and physical recovery require assessment; numerical correctness alone does not establish scene accuracy.'})
    result={'diagnostic_only':True,'real_scene_acceptance':False,'frames':len(scene.times),'cameras':1,'motion_per_frame':motion.tolist(),'true_final_moving_pose':true.tolist(),'independent_results':reports}
    (args.output/'accuracy_diagnostic.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
