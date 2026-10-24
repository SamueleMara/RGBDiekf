import tempfile,unittest
from pathlib import Path
import numpy as np
from numpy.testing import assert_allclose
from grouped_fixtures import GroupScene
from pointwise_rgbd.group_temporal_validation import solve_blocks,fit_temporal_raw,score_temporal_raw
from pointwise_rgbd.group_partition_validation import build_evidence_index,group_evidence,validate_group
from pointwise_rgbd.phase1_grouped_controller import run_phase1
from pointwise_rgbd.group_process import load_group_noise
from pointwise_rgbd.config import load_config
from rgbd_common.backends.torch_cuda import Backend


class TemporalValidationTests(unittest.TestCase):
    def test_sensor_boundary_split_does_not_weld_fixture_bodies(self):
        scene=GroupScene(frames=10,motion='boundary');c=load_config()
        with tempfile.TemporaryDirectory() as root:
            p,_=run_phase1(scene,c,load_group_noise(),Backend('cpu'),root,target_occupancy=1000)
            rows=group_evidence(build_evidence_index(root,p,10),0,scene);report=validate_group(p,0,rows,1)
            self.assertEqual(report['status'],'APPROVED_SPLIT')
            # Fixture truth is used ONLY here to detect a deceptively good
            # conditional likelihood with a physically incorrect boundary.
            truth=p.source[:,2]%scene.cameras[0].width>=44
            for label in (0,1):self.assertEqual(len(np.unique(truth[report['membership_labels']==label])),1,'A supposedly accepted rigid patch still straddles the test joint')

    def test_block_covariance_cross_correlation_and_logdet_match_dense_reference(self):
        rng=np.random.default_rng(104);size=[6,12,12,12];D=[];C=[]
        for d in size:
            x=rng.normal(size=(d,d));D.append(x@x.T+np.eye(d)*5)
        for i in range(1,len(size)):C.append(rng.normal(size=(size[i],size[i-1]))*.05)
        offsets=np.r_[0,np.cumsum(size)];dense=np.zeros((sum(size),sum(size)))
        for i in range(len(size)):
            a=slice(offsets[i],offsets[i+1]);dense[a,a]=D[i]
            if i:
                b=slice(offsets[i-1],offsets[i]);dense[a,b]=C[i-1];dense[b,a]=C[i-1].T
        rhs=[rng.normal(size=d) for d in size];result=solve_blocks(D,C,rhs);cov=np.linalg.inv(dense);mean=np.linalg.solve(dense,np.concatenate(rhs))
        for i in range(len(size)):
            a=slice(offsets[i],offsets[i+1]);assert_allclose(result['mean'][i],mean[a],atol=2e-15)
            assert_allclose(result['covariance'][i],cov[a,a],atol=2e-15)
            if i<len(size)-1:assert_allclose(result['cross_next'][i],cov[a,offsets[i+1]:offsets[i+2]],atol=2e-15)
        self.assertAlmostEqual(result['logdet'],np.linalg.slogdet(dense)[1],places=12)

    def test_static_wrong_group_does_not_gain_a_split_from_noise_free_sensor_data(self):
        scene=GroupScene(frames=8);c=load_config()
        with tempfile.TemporaryDirectory() as root:
            p,_=run_phase1(scene,c,load_group_noise(),Backend('cpu'),root,target_occupancy=1000)
            rows=group_evidence(build_evidence_index(root,p,8),0,scene);report=validate_group(p,0,rows,1)
            self.assertNotEqual(report['status'],'APPROVED_SPLIT')

    def test_low_instantaneous_support_is_propagated_in_temporal_gaussian(self):
        scene=GroupScene(frames=8,motion='translation');c=load_config()
        with tempfile.TemporaryDirectory() as root:
            p,_=run_phase1(scene,c,load_group_noise(),Backend('cpu'),root,target_occupancy=1000)
            rows=group_evidence(build_evidence_index(root,p,8),0,scene)
            for row in rows:
                if row['frame']==4:
                    for key in ('point_id','camera','pixel','z','R','rgb','xyz','covariance'):row[key]=row[key][:1]
            train=[np.ones(len(r['point_id']),bool) for r in rows]
            fit=fit_temporal_raw(p,rows,np.zeros(len(p.xyz),np.int64),1,train)
            self.assertEqual(fit['models'][0]['initial_twist_data_rank'],6)
            for P in fit['models'][0]['smoothed_covariance'][1:]:self.assertGreater(np.linalg.eigvalsh(P).min(),0)
            # No held-out data means no invented evidence, not a successful score.
            score=score_temporal_raw(p,rows,fit,[np.zeros(len(r['point_id']),bool) for r in rows])
            self.assertEqual(score['raw_count'],0);self.assertAlmostEqual(score['nll'],0)


if __name__=='__main__':unittest.main()
