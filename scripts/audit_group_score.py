"""Counterexample to the REMOVED experimental score, not a production prior."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from rgbd_common.config import load_config
from rgbd_common.dataloader import load_dataset
from rgbd_common.score import gaussian_nll


def main():
    p=argparse.ArgumentParser();p.add_argument('--input',required=True,type=Path);p.add_argument('--points',required=True,type=int);p.add_argument('--output',required=True,type=Path);a=p.parse_args()
    c=load_config();data=load_dataset(a.input)
    per_group=gaussian_nll(np.zeros(6),c.twist_covariance)+sum(gaussian_nll(np.zeros(12),c.q(dt)) for dt in np.diff(data.times))
    rows=[]
    for b in (1,2,4,8,9,10,12,16,20,32,64):
        delta=per_group+a.points*np.log((b+1)/b)+np.log((b+2)/b)
        rows.append({'old_groups':b,'new_groups':b+1,'score_delta_identity_group_split':float(delta)})
    # All points, poses, predictions and raw assignments are held physically
    # identical; only labels split into two identical-trajectory rigid groups.
    result={'purpose':'Exact algebraic counterexample to the removed implementation-invented structural prior; no estimation or data-dependent threshold','audited_model_is_production_default':False,'frames':len(data.times),'points':a.points,
            'per_group_zero_residual_motion_nll':per_group,'identity_split_counterexamples':rows,
            'interpretation':'Negative deltas reward splitting identical-motion labels with unchanged RGB-D predictions. Proper p(B) and uniform memberships alone do not make continuous MAP densities of different dimensions invariant or provide motion model evidence.',
            'status':'UNRESOLVED_TRANSDIMENSIONAL_MAP_MEASURE','required_resolution':'A justified model-selection measure (e.g. specified marginal evidence/ELBO) or explicitly authorized trajectory-aware structural prior; do not fabricate a magnitude or hide a group-count cap.'}
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
