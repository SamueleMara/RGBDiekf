import copy,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from numpy.testing import assert_array_equal
from pointwise_fixtures import PointScene,config
from rgbd_common.backends.torch_cuda import Backend
from rgbd_common.dataloader import RGBDFrame
from pointwise_rgbd.coverage import SurfaceIndex
from pointwise_rgbd.observations import initialize_all_frame_zero
from pointwise_rgbd.controller import run_exact_pointwise_discovery,match_all_frozen
from pointwise_rgbd.discovery import track_one_point
from pointwise_rgbd.incremental_coverage import euclidean_gate_radius


class IncrementalCoverageTests(unittest.TestCase):
    def test_same_surface_density_and_viewpoint_change_does_not_require_ownership(self):
        scene=PointScene(frames=3,cameras=1,hidden=False);c=config();b=Backend('cpu')
        for k in range(3):
            camera=copy.deepcopy(scene.cameras[0]);camera.world_to_camera[0,3]=k*.001
            rgb=np.zeros((48,64,3));depth=np.zeros((48,64));mask=np.zeros((48,64),bool)
            grid=[(y,x) for y in (23,25) for x in (31,33)] if k==0 else [(y,x) for y in range(23,26) for x in range(31,34)]
            for y,x in grid:mask[y,x]=True;depth[y,x]=1.;rgb[y,x]=[.7,.2,.1]
            scene.images[k][0]=RGBDFrame(rgb,depth,mask,camera,k,0)
        s,p,rgb,_=initialize_all_frame_zero(scene,c)
        from pointwise_rgbd.association import associate
        for k in range(3):
            frame=scene.observation(k,0);surface=SurfaceIndex(s,p,rgb,np.arange(len(s)))
            coverage=surface.match(frame,c,b)
            self.assertTrue(coverage.explained.all())
            tracking,_=associate(s,p,rgb,np.arange(len(s)),frame,c,b,np.full(64*48,-1,np.int64))
            self.assertLessEqual(len(tracking['pixel']),4)
            if k:self.assertEqual(int(coverage.explained.sum()),9)

    def test_separate_surface_and_incremental_fit_equals_pairwise_reference_without_old_reads(self):
        scene=PointScene();c=config();b=Backend('cpu')
        with tempfile.TemporaryDirectory() as root:
            store,owner=run_exact_pointwise_discovery(scene,c,b,root,max_events=0)
            baseline={(k,ci):owner.coverage.read(k,ci,0) for k in range(5) for ci in range(2)}
            for k in range(2,5):
                for ci in range(2):
                    r=baseline[k,ci];i=np.flatnonzero(r['pixel']==scene.hidden_pixels[k,ci])[0]
                    self.assertFalse(r['explained'][i])
            source=owner.first_unexplained(0);point=owner.allocate_id(store.n0)
            before=store.frozen_signature();owner.reserve(source,point)
            track,claims,color=track_one_point(point,source,scene,c,b,owner)
            path,record=store.save_new(point,track,source,color);owner.claim(claims,point_record=(point,path,record));store.append(path,record)
            # Prepare an exhaustive observation/new-point reference independently.
            reference={}
            for k in range(5):
                surface=SurfaceIndex(track['state'][k:k+1],track['covariance'][k:k+1],color[None],np.array([point]))
                for ci in range(2):reference[k,ci]=surface.match(scene.observation(k,ci),c,b)
            write=owner.coverage.store.write;interrupted=[False]
            def write_then_interrupt(key,**values):
                write(key,**values)
                if key[0].isdigit() and int(values['model_version'])==1 and not interrupted[0]:
                    interrupted[0]=True;raise RuntimeError('simulated interruption after atomic view commit')
            with patch.object(owner.coverage.store,'write',side_effect=write_then_interrupt):
                with self.assertRaisesRegex(RuntimeError,'simulated interruption'):
                    match_all_frozen(scene,store,owner,c,b)
            self.assertFalse(owner.coverage.completed(1))
            with patch.object(store,'frame',side_effect=AssertionError('old trajectories must not be read')),patch.object(scene,'observation',side_effect=AssertionError('raw images must not be reloaded')):
                result=match_all_frozen(scene,store,owner,c,b)
                self.assertEqual(result['old_trajectory_frame_reads'],0);self.assertEqual(result['raw_image_loads'],0)
                self.assertIsNone(match_all_frozen(scene,store,owner,c,b))
            for k in range(5):
                for ci in range(2):
                    actual=owner.coverage.read(k,ci,1);old=baseline[k,ci];new=reference[k,ci]
                    assert_array_equal(actual['explained'],old['explained']|new.explained)
                    assert_array_equal(actual['geometry_explained'],old['geometry_explained']|new.geometry_explained)
                    assert_array_equal(actual['point_id'][old['explained']],old['point_id'][old['explained']])
            self.assertTrue(interrupted[0])
            for ci in range(2):
                r=owner.coverage.read(source.frame,ci,1);i=np.flatnonzero(r['pixel']==scene.hidden_pixels[source.frame,ci])[0]
                self.assertTrue(r['explained'][i])
            store.assert_old_frozen(before)


    def test_new_incompatible_neighbors_do_not_erase_a_frozen_coverage_witness(self):
        import json
        from pointwise_rgbd.store import Ownership
        from pointwise_rgbd.incremental_coverage import cache_observations
        from types import SimpleNamespace
        scene=PointScene(frames=3,cameras=1,hidden=False);c=config();b=Backend('cpu')
        with tempfile.TemporaryDirectory() as root:
            owner=Ownership(scene,Path(root)/'owner.sqlite')
            s,p,rgb,_=initialize_all_frame_zero(scene,c);old=s.copy();old[:,0]+=.002
            baseline=[]
            for k in range(3):
                frame=scene.observation(k,0);result=SurfaceIndex(old,p,rgb,np.array([0])).match(frame,c,b)
                self.assertTrue(result.explained.all());owner.coverage.write(k,0,0,result)
                record=owner.coverage.read(k,0,0);baseline.append(record)
                cache_observations(owner.coverage,k,0,frame,c,record)
            owner.coverage.complete(0)
            # Each new point is closer than the valid old witness, but has
            # incompatible RGB. After k insertions a fresh top-k would drop the
            # old witness; monotone coverage correctly preserves it.
            state=np.repeat(s,3,axis=0);cov=np.repeat(p,3,axis=0)
            for n in range(c.neighbors+1):
                point=owner.allocate_id(1);path=Path(root)/f'candidate_{point}.npz'
                np.savez(path,state=state,covariance=cov)
                with owner.db:
                    owner.db.execute('INSERT INTO points VALUES (?,?,?)',(point,str(path),json.dumps({'color':[0.,0.,1.]})))
                    owner.set_meta('model_version',n+1)
                from pointwise_rgbd.incremental_coverage import update_insertions
                with patch.object(scene,'observation',side_effect=AssertionError('images reloaded')):
                    update_insertions(scene,owner,c,b)
                for k in range(3):
                    actual=owner.coverage.read(k,0,n+1)
                    assert_array_equal(actual['explained'],baseline[k]['explained'])
                    assert_array_equal(actual['point_id'],baseline[k]['point_id'])

    def test_spatial_prefilter_radius_is_necessary_not_a_novelty_threshold(self):
        rng=np.random.default_rng(18);gate=config().gate(3)
        for _ in range(40):
            a=rng.normal(size=(3,3));p=a@a.T
            a=rng.normal(size=(3,3));r=a@a.T
            direction=rng.normal(size=3);direction/=np.linalg.norm(direction)
            e=np.linalg.cholesky(2*(p+r))@direction*np.sqrt(gate)*.999
            radius=euclidean_gate_radius(p,np.linalg.eigvalsh(r)[-1],gate)
            self.assertLess(np.linalg.norm(e),radius)

    def test_uncommitted_partial_horizon_is_not_exposed(self):
        from pointwise_rgbd.coverage_store import CoverageIndex
        with tempfile.TemporaryDirectory() as root:
            index=CoverageIndex(Path(root)/'c.sqlite');index.complete(0)
            index.store.write('0:0',model_version=np.array(1),pixel=np.array([5]),explained=np.array([True]))
            with self.assertRaisesRegex(ValueError,'STALE'):index.read(0,0,0)
            index.complete(1,17);self.assertTrue(index.read(0,0,1)['explained'][0])
