import copy,tempfile,unittest
from pathlib import Path
import numpy as np
from numpy.testing import assert_array_equal
from grouped_fixtures import GroupScene
from pointwise_rgbd.config import load_config
from pointwise_rgbd.group_process import load_group_noise
from pointwise_rgbd.phase1_grouped_controller import run_phase1
from pointwise_rgbd.group_partition_validation import build_evidence_index,group_evidence,validate_group,propose_split,split_partition,predictive_score,fit_measured_rigid
from rgbd_common.backends.torch_cuda import Backend


class GroupBoundaryTests(unittest.TestCase):
    def test_wrong_initial_group_rejected_from_heldout_rgbd_and_both_groups_refit(self):
        scene=GroupScene(frames=10,motion='boundary');config=load_config();noise=load_group_noise();b=Backend('cpu')
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);partition,arrays=run_phase1(scene,config,noise,b,root/'v0',target_occupancy=1000)
            self.assertEqual(partition.count,1) # Deliberately erroneous computational group, no correct labels supplied.
            db=build_evidence_index(root/'v0',partition,len(scene.times));rows=group_evidence(db,0,scene)
            report=validate_group(partition,0,rows,1)
            self.assertEqual(report['status'],'APPROVED_SPLIT',str({key:value for key,value in report.items() if key!='membership_labels'}))
            new=split_partition(partition,0,report['membership_labels']);self.assertEqual(new.count,2);self.assertEqual(new.version,1)
            _,refit=run_phase1(scene,config,noise,b,root/'v1',partition=new)
            self.assertEqual(refit['xyz'].shape,arrays['xyz'].shape)
            for k in range(10):
                for group in range(2):
                    ids=np.flatnonzero(new.membership==group)
                    self.assertLess(np.abs(np.linalg.norm(refit['xyz'][k,ids]-refit['xyz'][k,ids[0]],axis=1)-np.linalg.norm(new.xyz[ids]-new.xyz[ids[0]],axis=1)).max(),1e-12)
            frames=report['proposal_frames'];proposal,_=propose_split(partition,0,rows,frames)
            changed=copy.deepcopy(rows)
            for row in changed:
                test=partition.duplicate_unit[row['point_id']]%2==0;row['xyz'][test]+=100
            after,_=propose_split(partition,0,changed,frames);assert_array_equal(proposal,after)
            val=report['member_holdout'];self.assertFalse(set(map(tuple,val['identical_evaluation_refs']))&set(map(tuple,val['training_refs'])))
    def test_training_proposal_does_not_read_heldout_measurement_values(self):
        scene=GroupScene(frames=10,motion='boundary');config=load_config();b=Backend('cpu')
        with tempfile.TemporaryDirectory() as root:
            partition,_=run_phase1(scene,config,load_group_noise(),b,root,target_occupancy=1000)
            rows=group_evidence(build_evidence_index(root,partition,10),0,scene);frames=[1,2,3,4];before={};after={}
            propose_split(partition,0,rows,frames,before)
            changed=copy.deepcopy(rows)
            for row in changed:
                heldout=partition.duplicate_unit[row['point_id']]%2==0;row['xyz'][heldout]+=1000
            propose_split(partition,0,changed,frames,after)
            assert_array_equal(before['training_features'],after['training_features'])
            assert_array_equal(before['proposed_membership'],after['proposed_membership'])
    def test_multicamera_same_surface_duplicates_have_linked_holdout_units(self):
        from pointwise_rgbd.surface_supernodes import build_partition
        scene=GroupScene(frames=3,cameras=2);_,partition=build_partition(scene,'first',load_config(),target_occupancy=1000)
        unit=partition.duplicate_unit
        counts=np.bincount(unit)
        self.assertTrue((counts==2).all())
        for value in np.unique(unit):
            camera=partition.source[unit==value,1]
            assert_array_equal(np.sort(camera),[0,1])
            self.assertEqual(len(set((unit[unit==value]%2).tolist())),1)

    def test_shared_pose_predictive_score_matches_dense_reference(self):
        p=np.array([[-.2,-.1,1.],[-.2,.1,1.],[.2,-.1,1.],[.2,.1,1.]])
        cov=np.repeat(np.diag([.00001,.00002,.0001])[None],4,axis=0);y=p+np.array([.01,-.002,.02])
        fit=fit_measured_rigid(p,y,cov);obs=y+np.array([.002,.001,-.003])
        score=predictive_score(p,obs,cov,fit)
        from scipy.linalg import block_diag
        from rgbd_common.lie import skew
        xyz=p@fit['pose'][:3,:3].T+fit['pose'][:3,3];j=np.vstack([np.c_[np.eye(3),-skew(x)] for x in xyz]);s=block_diag(*cov)+j@fit['covariance']@j.T;e=(obs-xyz).ravel()
        dense=.5*(e@np.linalg.solve(s,e)+np.linalg.slogdet(s)[1]+len(e)*np.log(2*np.pi))
        self.assertAlmostEqual(score['nll'],dense,places=9)


if __name__=='__main__':unittest.main()
