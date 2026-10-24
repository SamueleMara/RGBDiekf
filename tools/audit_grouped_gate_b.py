#!/usr/bin/env python3
"""Preserve the actual failing synthetic RGB-D boundary gate; never call it PASS."""
import argparse,json,sys
from pathlib import Path
import numpy as np
REPO=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO));sys.path.insert(0,str(REPO/'tests'))
from grouped_fixtures import GroupScene
from pointwise_rgbd.config import load_config
from pointwise_rgbd.group_process import load_group_noise
from pointwise_rgbd.phase1_grouped_controller import run_phase1
from pointwise_rgbd.group_partition_validation import build_evidence_index,group_evidence,propose_split,heldout_comparison,fit_measured_rigid
from pointwise_rgbd.store import json_write
from rgbd_common.backends.torch_cuda import Backend


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args();root=args.output
    if root.exists() and any(root.iterdir()):parser.error('Use a fresh output; previous evidence is preserved')
    root.mkdir(parents=True);scene=GroupScene(frames=10,motion='boundary');config=load_config()
    partition,arrays=run_phase1(scene,config,load_group_noise(),Backend('cpu'),root/'phase1',target_occupancy=1000)
    for k in range(10):
        frame=scene.observation(k,0);np.savez_compressed(root/f'actual_synthetic_rgbd_{k:03d}.npz',rgb=frame.rgb,depth=frame.depth,mask=frame.mask,valid=frame.valid,intrinsics=frame.camera.intrinsics,world_to_camera=frame.camera.world_to_camera,original_frame=np.array(scene.frame_ids[k]),time_seconds=np.array(scene.times[k]))
    evidence=group_evidence(build_evidence_index(root/'phase1',partition,10),0,scene)
    proposal_frames=sorted({r['frame'] for r in evidence})[:len(evidence)//2];details={}
    assignment,reason=propose_split(partition,0,evidence,proposal_frames,details)
    if 'proposed_membership' in details:
        np.savez_compressed(root/'rejected_training_proposal.npz',**details)
        diagnostic=heldout_comparison(partition,evidence,details['proposed_membership'])
        json_write(root/'heldout_invalid_proposal_diagnostic_only.json',diagnostic)
        support=[]
        for row in evidence:
            parts=[]
            for label in (0,1):
                use=(details['proposed_membership'][row['point_id']]==label)&(partition.duplicate_unit[row['point_id']]%2==1)
                fit=fit_measured_rigid(partition.xyz[row['point_id'][use]],row['xyz'][use],row['covariance'][use])
                parts.append({'label':label,'training_raw_samples':int(use.sum()),'pose_rank':fit['rank']})
            support.append({'frame':row['frame'],'actual_accepted_raw_samples':len(row['point_id']),'proposed_subpatch_training_support':parts})
        json_write(root/'actual_support_by_frame.json',support)
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,ax=plt.subplots(1,2,figsize=(12,5))
        frame=scene.observation(0,0);ax[0].imshow(frame.rgb);ax[0].set_title('Actual synthetic endpoint RGB (estimator input)')
        xyz=partition.xyz;ax[1].scatter(xyz[:,0],xyz[:,1],c=details['proposed_membership'],s=20,cmap='coolwarm');ax[1].set(xlabel='Canonical world X (m)',ylabel='Canonical world Y (m)',title='Rejected training-residual labels: disconnected')
        fig.tight_layout();fig.savefig(root/'rejected_proposal.png',dpi=160);plt.close(fig)
    report={'status':'GATE_B_FAILED' if assignment is None else 'UNEXPECTED_PROPOSAL_REQUIRES_REVIEW','reason':reason,'initial_supernodes':partition.count,'raw_endpoint_points':len(partition.xyz),'frames':10,'proposal_frames':proposal_frames,'subpatch_sizes':details.get('subpatch_sizes'),'subpatch_connected_components':details.get('subpatch_connected_components'),'heldout_used_for_proposal':False,'group_generated_positions_used_for_split_evidence':False,'accepted_splits':0,'full_sequence_refit_after_split':False,'ground_truth_labels_read_by_estimator':False,'input_kind':'explicit synthetic RGB-D sensor regression, not the real fridge dataset','narrow_patch':'Generate connected graph cuts from training residual evidence rather than global two-means labels; keep current held-out/covariance/complexity tests unchanged.'}
    json_write(root/'gate_b_failure.json',report);print(json.dumps(report,indent=2));return 1 if assignment is None else 2


if __name__=='__main__':raise SystemExit(main())
