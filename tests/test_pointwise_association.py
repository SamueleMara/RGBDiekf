import tempfile,unittest,copy,hashlib
from pathlib import Path
from unittest.mock import patch
import numpy as np
from numpy.testing import assert_allclose,assert_array_equal
import torch
from rgbd_common.backends.torch_cuda import Backend
from pointwise_rgbd.observations import initialize_all_frame_zero
from pointwise_rgbd.association import associate,exclusive_assignment
from pointwise_rgbd.controller import run_exact_pointwise_discovery,match_all_frozen
from pointwise_rgbd.filter_iekf import correct
from pointwise_rgbd.store import TrackStore,Ownership
from pointwise_fixtures import PointScene,config


class PointwiseAssociationTests(unittest.TestCase):
    def test_all_frame_zero_samples_preserved_and_background_invalid_invariance(self):
        scene=PointScene(frames=3,hidden=False);c=config();backend=Backend('cpu')
        original=initialize_all_frame_zero(scene,c)
        self.assertEqual(len(original[0]),sum(int(scene.observation(0,i).valid.sum()) for i in range(2)))
        frame=scene.observation(1,0);owners=np.full(64*48,-1,np.int64)
        a,stats=associate(original[0],original[1],original[2],np.arange(len(original[0])),frame,c,backend,owners)
        changed=copy.deepcopy(scene);rng=np.random.default_rng(27)
        for row in changed.images:
            for image in row:
                image.raw_depth_validity=image.raw_valid.copy();background=~image.mask
                image.rgb[background]=rng.uniform(0,1,(background.sum(),3));image.depth[background]=rng.uniform(.1,1e5,background.sum())
                image.depth[image.mask&~image.raw_depth_validity]=np.nan
        after=initialize_all_frame_zero(changed,c)
        for x,y in zip(original,after):assert_array_equal(x,y)
        b,_=associate(after[0],after[1],after[2],np.arange(len(after[0])),changed.observation(1,0),c,backend,owners)
        for key in ('pixel','point','z','R'):assert_array_equal(a[key],b[key])
        ids=original[3];self.assertEqual(len(np.unique(ids,axis=0)),len(ids))
    def test_raw_exclusive_assignment_and_unowned_geometry_are_exact(self):
        point=np.array([0,0,1,1,2]);pixel=np.array([5,6,5,7,7]);cost=np.array([1.,2.,.5,3.,.4]);ids=np.array([100,101,102])
        selected=exclusive_assignment(point,pixel,cost,ids)
        self.assertEqual(len(set(pixel[selected])),len(selected));self.assertEqual(len(set(point[selected])),len(selected))
        scene=PointScene(frames=3,hidden=False);c=config();backend=Backend('cpu');state,p,rgb,source=initialize_all_frame_zero(scene,c)
        owner=np.full(64*48,-1,np.int64);owner[np.flatnonzero(scene.observation(1,0).valid.ravel())]=900
        m,_=associate(state,p,rgb,np.arange(len(state)),scene.observation(1,0),c,backend,owner)
        self.assertEqual(len(m['pixel']),0)
    def test_static_translation_rotating_curved_and_depth_holes(self):
        for motion in ('static','translation','rotation'):
            with self.subTest(motion=motion),tempfile.TemporaryDirectory() as root:
                scene=PointScene(frames=7,motion=motion,hidden=False)
                if motion=='rotation':
                    for frame in scene.images[3]:frame.depth[:]=0
                store,owner=run_exact_pointwise_discovery(scene,config(),Backend('cpu'),root,max_events=0)
                x=store.arrays['state'][:,:,0:3]
                self.assertTrue(np.isfinite(x).all());self.assertGreater(np.linalg.eigvalsh(store.arrays['covariance']).min(),0)
                if motion=='static':self.assertLess(np.linalg.norm(x[-1,0]-x[0,0]),.01)
                else:self.assertGreater(np.linalg.norm(x[-1,0]-x[0,0]),.015)
                if motion=='rotation':self.assertFalse(store.arrays['observed'][3].any())
    def test_failure_controller_skips_same_version_and_inserts_only_completed_fit(self):
        from pointwise_rgbd.discovery import track_one_point
        calls=[]
        def fail_first(point,*args,**kwargs):
            calls.append(point)
            if len(calls)==1:raise ArithmeticError('deliberate SPD failure fixture')
            return track_one_point(point,*args,**kwargs)
        with tempfile.TemporaryDirectory() as root,patch('pointwise_rgbd.controller.track_one_point',side_effect=fail_first):
            store,owner=run_exact_pointwise_discovery(PointScene(),config(),Backend('cpu'),root,max_events=1)
            self.assertEqual(len(calls),2);self.assertNotEqual(calls[0],calls[1]);self.assertEqual(len(store.added),1)
            self.assertEqual(store.added[0]['point_id'],calls[-1]);self.assertTrue(list(Path(root).glob('discovery/events/*/failure_anchor.png')))
    def test_cached_version_scan_equals_brute_force_with_frozen_geometry(self):
        import shutil
        scene=PointScene();c=config();b=Backend('cpu')
        with tempfile.TemporaryDirectory() as root:
            store,owner=run_exact_pointwise_discovery(scene,c,b,root,max_events=1)
            before=[owner.get(k,ci).copy() for k in range(5) for ci in range(2)]
            fast=owner.first_unclaimed(owner.meta('model_version',0))
            with owner.db:owner.set_meta('matched_model_version',-1)
            match_all_frozen(scene,store,owner,c,b)
            brute=owner.first_unclaimed(owner.meta('model_version',0))
            self.assertEqual(fast,brute)
            after=[owner.get(k,ci).copy() for k in range(5) for ci in range(2)]
            for x,y in zip(before,after):assert_array_equal(x,y)
    def test_file_backed_history_has_only_per_point_six_state_covariances(self):
        scene=PointScene();n=10000
        with tempfile.TemporaryDirectory() as root:
            path=Path(root)/'initial_forward/trajectory_store'
            store=TrackStore(path,scene,n,(np.zeros((n,3)),np.zeros((n,3),np.int64)))
            self.assertIsInstance(store.arrays['covariance'],np.memmap)
            self.assertEqual(store.arrays['covariance'].shape,(5,n,6,6))
            self.assertLess(sum(p.stat().st_size for p in path.glob('*')),5*n*400)


@unittest.skipUnless(torch.cuda.is_available(),'CUDA hardware required')
class PointwiseCUDATests(unittest.TestCase):
    def test_cuda_candidates_information_correction_and_covariance_equal_cpu(self):
        scene=PointScene(frames=3,motion='translation',hidden=False);c=config()
        state,p,rgb,source=initialize_all_frame_zero(scene,c);state[:,:3]+=[.002,-.001,.002]
        results=[]
        for device in ('cpu','cuda'):
            backend=Backend(device);m,stats=associate(state,p,rgb,np.arange(len(state)),scene.observation(1,0),c,backend,np.full(64*48,-1,np.int64))
            s,cov,report=correct(backend.tensor(state),backend.tensor(p),[m],c)
            results.append((m['pixel'],s.cpu().numpy(),cov.cpu().numpy()))
        assert_array_equal(results[0][0],results[1][0]);assert_allclose(results[0][1],results[1][1],atol=1e-12);assert_allclose(results[0][2],results[1][2],atol=1e-12)


if __name__=='__main__':unittest.main()
