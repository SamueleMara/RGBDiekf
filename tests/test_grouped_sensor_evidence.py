import json,tempfile,unittest
from pathlib import Path
import numpy as np
from numpy.testing import assert_allclose,assert_array_equal
from grouped_fixtures import GroupScene
from pointwise_rgbd.config import load_config
from pointwise_rgbd.group_process import load_group_noise
from pointwise_rgbd.phase1_grouped_controller import run_phase1
from pointwise_rgbd.group_evidence_audit import (read_factors,load_cached,replay_candidates,projection_numpy,group_information)
from pointwise_rgbd.group_visualization import export_viewer
from pointwise_rgbd.observations import samples
from pointwise_rgbd.association import associate
from rgbd_common.backends.torch_cuda import Backend


class SensorEvidenceTests(unittest.TestCase):
    def test_joint_sensor_space_predictive_score_matches_dense_small_reference(self):
        import torch
        from scipy.linalg import block_diag
        from pointwise_rgbd.group_predictive import heldout_uvz_score
        from pointwise_rgbd.projection import project
        from pointwise_rgbd.group_pose import point_jacobian
        b=Backend('cpu');scene=GroupScene(frames=2);camera=scene.cameras[0]
        p=b.tensor([[-.1,-.03,1.],[.05,-.02,1.],[.02,.06,1.]])
        h,j=project(p,camera);H=j[:,:,:3]@point_jacobian(p);R=np.repeat(np.diag([.1,.1,.0001])[None],3,axis=0)
        P=np.diag([.0001,.0001,.0001,.00001,.00001,.00001]);error=np.array([[.1,.2,.002],[.1,-.1,.003],[.2,0,.001]])
        match={'camera':camera,'point':b.tensor(np.arange(3),torch.long),'z':h+b.tensor(error),'R':b.tensor(R)}
        score=heldout_uvz_score(b.tensor(np.eye(4)[None]),b.tensor(P[None]),p,b.tensor(np.zeros(3,np.int64),torch.long),[match],b)
        dense=block_diag(*R)+H.numpy().reshape(-1,6)@P@H.numpy().reshape(-1,6).T;e=error.ravel()
        expected=.5*(e@np.linalg.solve(dense,e)+np.linalg.slogdet(dense)[1]+9*np.log(2*np.pi))
        self.assertAlmostEqual(score['nll_per_group'][0],expected,places=9)

    def test_read_only_actual_sensor_export_and_source_reprojection(self):
        scene=GroupScene(frames=4,motion='translation',cameras=2);c=load_config();b=Backend('cpu')
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);p,a=run_phase1(scene,c,load_group_noise(),b,root/'cache')
            before={f.name:(f.stat().st_size,f.stat().st_mtime_ns) for f in (root/'cache').iterdir() if f.is_file()}
            loaded,_=load_cached(root/'cache');self.assertEqual(loaded.digest(),p.digest())
            manifest=export_viewer(root/'diagnostics',scene,p,a,root/'cache',c,replay=True)
            self.assertNotIn('measured_bounds',manifest);self.assertIn('canonical_focus_bounds',manifest)
            for k in range(4):
                for ci in range(2):
                    prefix=root/'diagnostics/viewer/frames'/f'{k:06d}.c{ci:03d}'
                    raw=np.fromfile(str(prefix)+'.raw.f32',dtype='<f4').reshape(-1,3)
                    assert_allclose(raw,samples(scene.observation(k,ci),c).xyz,atol=6e-8,rtol=6e-8)
                    self.assertLess(max(manifest['frames'][k]['cameras'][ci]['roundtrip_max_abs_uvz']),1e-12)
            self.assertEqual(before,{f.name:(f.stat().st_size,f.stat().st_mtime_ns) for f in (root/'cache').iterdir() if f.is_file()})

    def test_candidate_replay_matches_baseline_family_without_changing_matcher(self):
        scene=GroupScene(frames=3);frame=scene.observation(1,0);c=load_config();b=Backend('cpu');raw=samples(frame,c)
        xyz=raw.xyz+np.array([.0001,-.0002,.0003]);state=np.c_[xyz,np.zeros_like(xyz)]
        cov=np.repeat(np.eye(3)[None]*1e-6,len(xyz),axis=0);full=np.zeros((len(xyz),6,6));full[:,:3,:3]=cov
        match,_=associate(state,full,raw.rgb,np.arange(len(xyz)),frame,c,b,np.full(frame.depth.size,-1))
        reason,candidate=replay_candidates(xyz,cov,raw.rgb,raw,c)
        selected=np.flatnonzero(candidate['reason']==0);a=selected[np.argsort(candidate['point'][selected])];order=np.argsort(match['point'].numpy())
        assert_array_equal(candidate['point'][a],match['point'].numpy()[order]);assert_array_equal(candidate['pixel'][a],match['pixel'][order])
        assert_allclose(candidate['d2'][a],match['d2'][order],rtol=1e-12,atol=1e-12)
        self.assertEqual(len(reason),len(xyz));self.assertTrue(np.isfinite(candidate['S']).all())

    def test_angular_information_uses_radius_and_exports_units_not_rank_claim(self):
        scene=GroupScene(frames=3);c=load_config();b=Backend('cpu')
        with tempfile.TemporaryDirectory() as temporary:
            p,a=run_phase1(scene,c,load_group_noise(),b,temporary,target_occupancy=1000)
            d=group_information(p,a['group_poses'][1],a['covariance'][1],read_factors(temporary,1),scene.cameras)
            self.assertEqual(d['normalized_information_eigenvalues'].shape,(1,6));self.assertTrue((d['angular_schur_information']>=-1e-10).all())
            self.assertEqual(d['distinct_cameras'][0],1);self.assertEqual(d['distinct_points'][0],200)
            self.assertIn('NOT_PRECISION_CERTIFIED',d['status'][0]);self.assertTrue((d['angular_std_rad']>0).all())


if __name__=='__main__':unittest.main()
