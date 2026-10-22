import tempfile,unittest,copy
from pathlib import Path
import numpy as np
import torch
from numpy.testing import assert_allclose,assert_array_equal
from pointwise_rgbd.coverage import SurfaceIndex,appearance_terms
from pointwise_rgbd.observations import initialize_all_frame_zero,samples
from pointwise_fixtures import PointScene,config
from rgbd_common.backends.torch_cuda import Backend


class CoverageTests(unittest.TestCase):
    def test_background_and_raw_invalid_values_cannot_change_surface_coverage(self):
        scene=PointScene(frames=3,hidden=False);c=config();s,p,rgb,_=initialize_all_frame_zero(scene,c)
        image=scene.observation(1,0);image.mask[0,0]=True;image.raw_depth_validity=image.raw_valid.copy()
        surface=SurfaceIndex(s,p,rgb,np.arange(len(s)));b=Backend('cpu');before=surface.match(image,c,b)
        changed=copy.deepcopy(image);unused=~image.valid;rng=np.random.default_rng(7)
        changed.depth[unused]=rng.uniform(.1,1e6,int(unused.sum()));changed.rgb[unused]=rng.uniform(0,1,(int(unused.sum()),3))
        after=surface.match(changed,c,b)
        for name in ('pixel','explained','geometry_explained','point_id'):assert_array_equal(getattr(before,name),getattr(after,name))
    def test_unknown_cross_covariance_upper_bound_is_not_fake_independence(self):
        rng=np.random.default_rng(913)
        for _ in range(12):
            a=rng.normal(size=(6,6));joint=a@a.T;p=joint[:3,:3];r=joint[3:,3:];cross=joint[:3,3:]
            actual=p+r-cross-cross.T;bound=2*(p+r)
            self.assertGreaterEqual(np.linalg.eigvalsh(bound-actual).min(),-1e-12)
    def test_controlled_legacy_ablation_exposes_only_ownership_restriction(self):
        from pointwise_rgbd.legacy_coverage_audit import nonexclusive_legacy
        from pointwise_rgbd.association import associate
        scene=PointScene(frames=3,hidden=False);c=config();image=scene.observation(1,0)
        y,x=np.argwhere(image.valid)[0];image.mask[y,x+1]=True;image.depth[y,x+1]=image.depth[y,x];image.rgb[y,x+1]=image.rgb[y,x]
        s,p,rgb,_=initialize_all_frame_zero(scene,c);s=s[:1];p=p[:1];rgb=rgb[:1];owner=np.full(image.depth.size,-1,np.int64);owner[y*64+x]=17
        surface=SurfaceIndex(s,p,rgb,np.array([17]));raw=samples(image,c);b=Backend('cpu')
        tracking,_=associate(s,p,rgb,np.array([17]),image,c,b,owner)
        self.assertEqual(len(tracking['pixel']),0)
        coverage,same_ties,witness,_=nonexclusive_legacy(surface,image,raw,owner,c,b)
        self.assertTrue(coverage.all());self.assertTrue(same_ties.all());self.assertTrue(witness.all())
    def test_coverage_does_not_claim_pixels_and_discovery_uses_coverage(self):
        from pointwise_rgbd.store import Ownership
        scene=PointScene(frames=3,hidden=False);c=config();state,p,rgb,_=initialize_all_frame_zero(scene,c)
        with tempfile.TemporaryDirectory() as root:
            owner=Ownership(scene,Path(root)/'owners.sqlite');surface=SurfaceIndex(state,p,rgb,np.arange(len(state)))
            for k in range(3):
                for ci in range(2):
                    result=surface.match(scene.observation(k,ci),c,Backend('cpu'));before=owner.get(k,ci).copy()
                    owner.coverage.write(k,ci,0,result);assert_array_equal(owner.get(k,ci),before)
            owner.coverage.complete(0)
            self.assertIsNone(owner.first_unexplained(0));self.assertIsNotNone(owner.first_unclaimed(0))
            with self.assertRaisesRegex(ValueError,'STALE'):owner.coverage.read(0,0,1)
            frame=scene.observation(0,0);pixel=int(np.flatnonzero(frame.valid.ravel())[0])
            with self.assertRaisesRegex(ValueError,'coverage'):owner.claim([(0,0,np.array([pixel]),np.array([0]))],kind=2)
    def test_owned_but_unexplained_measurement_is_reported_without_reassimilation(self):
        from pointwise_rgbd.store import Ownership
        from pointwise_rgbd.coverage_store import FrozenMeasurementConflict
        from pointwise_rgbd.state6 import ObservationRef
        scene=PointScene(frames=3,hidden=False);c=config();state,p,rgb,_=initialize_all_frame_zero(scene,c)
        with tempfile.TemporaryDirectory() as root:
            owner=Ownership(scene,Path(root)/'owners.sqlite');surface=SurfaceIndex(state,p,rgb,np.arange(len(state)))
            for k in range(3):
                for ci in range(2):owner.coverage.write(k,ci,0,surface.match(scene.observation(k,ci),c,Backend('cpu')))
            record=owner.coverage.read(2,0,0);record['explained'][:]=False;owner.coverage.store.write('2:0',**record)
            pixel=int(record['pixel'][0]);owner.claim([(2,0,np.array([pixel]),np.array([17]))])
            self.assertIsNone(owner.first_unexplained(0))
            self.assertFalse(owner.coverage.read(2,0,0)['explained'][0])
            self.assertEqual(int(owner.tracking_owners(2,0)[pixel]),17)
    def test_cached_discovery_never_calls_initial_optimizer_or_changes_history(self):
        import json,hashlib
        from unittest.mock import patch
        from pointwise_rgbd.controller import run_exact_pointwise_discovery
        from pointwise_rgbd.frozen import FrozenCache
        scene=PointScene();c=config();b=Backend('cpu')
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);original,owners=run_exact_pointwise_discovery(scene,c,b,root/'original',max_events=0)
            (root/'original/run_manifest.json').write_text(json.dumps({'config':c.manifest()}))
            before={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in original.directory.iterdir() if p.is_file()}
            frozen=FrozenCache(root/'original',root/'next/discovery/tracks')
            with patch('pointwise_rgbd.controller.initial_forward',side_effect=AssertionError('must never refit old points')):
                store,owner=run_exact_pointwise_discovery(scene,c,b,root/'next',max_events=1,cached_store=frozen)
            self.assertEqual(len(store.added),1);self.assertEqual(owner.meta('initial_invocations'),1)
            self.assertEqual(before,{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in original.directory.iterdir() if p.is_file()})
    def test_audited_coverage_import_preserves_means_and_skips_both_optimizers(self):
        import json,hashlib
        from unittest.mock import patch
        from test_data_loading import real_scene
        from pointwise_rgbd.dataset import Dataset
        from pointwise_rgbd.controller import run_exact_pointwise_discovery
        from pointwise_rgbd.frozen import FrozenCache
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);scene=root/'scene';scene.mkdir();real_scene(scene);data=Dataset(scene);c=config();b=Backend('cpu')
            store,owner=run_exact_pointwise_discovery(data,c,b,root/'old',max_events=0)
            source_manifest={'config':c.manifest()};(root/'old/run_manifest.json').write_text(json.dumps(source_manifest))
            audit=root/'audit';(audit/'raw_classification').mkdir(parents=True)
            for k in range(3):
                state,p,rgb,ids,_=store.frame(k);surface=SurfaceIndex(state,p,rgb,ids)
                for ci in range(2):
                    result=surface.match(data.observation(k,ci),c,b)
                    np.savez(audit/'raw_classification'/f'{k:06d}_{ci:03d}.npz',pixel=result.pixel,appearance_explained=result.explained,geometry_explained=result.geometry_explained,appearance_point_id=result.point_id,geometry_point_id=result.geometry_point_id)
            (audit/'summary.json').write_text(json.dumps({'status':'AUDIT_COMPLETE','all_cached_frames':True,'optimizer_calls':0,'cached_SOLVED':store.count,'frames_checked':3,'cameras_checked':2}))
            kernel=Path(__file__).resolve().parents[1]/'pointwise_rgbd/coverage.py'
            (audit/'manifest.json').write_text(json.dumps({'cache':str((root/'old').resolve()),'source_hashes':{'pointwise_rgbd/coverage.py':hashlib.sha256(kernel.read_bytes()).hexdigest()}}))
            (audit/'audited_model.json').write_text(json.dumps({'source_run_manifest_sha256':hashlib.sha256((root/'old/run_manifest.json').read_bytes()).hexdigest()}))
            frozen=FrozenCache(root/'old',root/'new/discovery/tracks')
            with patch('pointwise_rgbd.controller.initial_forward',side_effect=AssertionError('no initial refit')),patch('pointwise_rgbd.controller.track_one_point',side_effect=AssertionError('no new fit in coverage check')):
                copied,new_owner=run_exact_pointwise_discovery(data,c,b,root/'new',max_events=0,cached_store=frozen,coverage_audit=audit)
            self.assertTrue(new_owner.coverage.completed(0));self.assertEqual(copied.count,store.count)
            final=json.loads((root/'new/final/summary.json').read_text());self.assertEqual(final['initial_optimizer_calls_this_invocation'],0);self.assertEqual(final['point_optimizer_calls_this_invocation'],0)
    def test_complete_frame_zero_is_explained_without_any_owner_input(self):
        scene=PointScene();c=config();state,p,rgb,_=initialize_all_frame_zero(scene,c)
        surface=SurfaceIndex(state,p,rgb,np.arange(len(state)))
        for ci in range(2):
            result=surface.match(scene.observation(0,ci),c,Backend('cpu'))
            self.assertTrue(result.explained.all());assert_allclose(result.nearest_distance_m,0,atol=1e-14)
    def test_many_observations_can_share_one_surface_point_but_tracker_is_exclusive(self):
        scene=PointScene(frames=3,hidden=False);c=config();image=scene.observation(1,0)
        y,x=np.argwhere(image.valid)[0];image.mask[y,x+1]=True;image.depth[y,x+1]=image.depth[y,x];image.rgb[y,x+1]=image.rgb[y,x]
        state,p,rgb,_=initialize_all_frame_zero(scene,c);state=state[:1];p=p[:1];rgb=rgb[:1]
        result=SurfaceIndex(state,p,rgb,np.array([17])).match(image,c,Backend('cpu'))
        self.assertEqual(int(result.explained.sum()),2);assert_array_equal(result.point_id,[17,17])
        from pointwise_rgbd.association import associate
        match,_=associate(state,p,rgb,np.array([17]),image,c,Backend('cpu'),np.full(image.depth.size,-1,np.int64))
        self.assertLessEqual(len(match['pixel']),1)
    def test_rgb_cosine_and_brightness_are_one_statistical_color_distance(self):
        a=torch.tensor([[.1,.1,.1],[.8,0,0],[0,0,0]],dtype=torch.float64)
        b=torch.tensor([[.9,.9,.9],[0,.8,0],[.2,.2,.2]],dtype=torch.float64)
        covariance=torch.eye(3,dtype=torch.float64)*.01
        d2,cosine,_,angular,brightness=appearance_terms(a,b,covariance)
        assert_allclose(d2,angular+brightness,atol=1e-12);self.assertAlmostEqual(float(cosine[0]),1.)
        self.assertGreater(float(brightness[0]),config().gate(3));self.assertEqual(float(angular[0]),0.)
        self.assertTrue(torch.isnan(cosine[2]));self.assertTrue(torch.isfinite(d2).all())
    def test_color_mismatch_rejected_despite_same_geometry_and_gray_direction(self):
        scene=PointScene(frames=3,hidden=False);c=config();s,p,rgb,_=initialize_all_frame_zero(scene,c)
        image=scene.observation(0,0);image.rgb[image.valid]=.9;rgb[:]=.1
        result=SurfaceIndex(s,p,rgb,np.arange(len(s))).match(image,c,Backend('cpu'))
        self.assertTrue(result.geometry_explained.all());self.assertFalse(result.explained.any())
        assert_allclose(result.geometry_rgb_cosine,1)
    def test_world_translation_changes_neither_correspondence_nor_scores(self):
        scene=PointScene(frames=3,hidden=False);c=config();s,p,rgb,_=initialize_all_frame_zero(scene,c);ids=np.arange(len(s));b=Backend('cpu')
        first=SurfaceIndex(s,p,rgb,ids).match(scene.observation(0,0),c,b)
        shifted=copy.deepcopy(scene.observation(0,0));offset=np.array([123.,-55.,29.]);ss=s.copy();ss[:,:3]+=offset
        shifted.camera.world_to_camera[:3,3]-=shifted.camera.world_to_camera[:3,:3]@offset
        second=SurfaceIndex(ss,p,rgb,ids).match(shifted,c,b)
        assert_array_equal(first.explained,second.explained);assert_array_equal(first.point_id,second.point_id)
        assert_allclose(first.nearest_distance_m,second.nearest_distance_m,atol=1e-12)
    def test_depth_occlusion_invalid_masks_and_invalid_covariance_do_not_explain(self):
        scene=PointScene(frames=3,hidden=False);c=config();s,p,rgb,_=initialize_all_frame_zero(scene,c)
        s[:,2]+=1
        result=SurfaceIndex(s,p,rgb,np.arange(len(s))).match(scene.observation(0,0),c,Backend('cpu'))
        self.assertFalse(result.explained.any())
        s,p,rgb,_=initialize_all_frame_zero(scene,c);p[:,:3,:3]=-np.eye(3)
        result=SurfaceIndex(s,p,rgb,np.arange(len(s))).match(scene.observation(0,0),c,Backend('cpu'))
        self.assertFalse(result.explained.any());self.assertGreater(result.stats['invalid_uncertainty_edges'],0)
        image=scene.observation(0,0);image.mask[:]=False
        self.assertEqual(len(SurfaceIndex(s,p,rgb,np.arange(len(s))).match(image,c,Backend('cpu')).pixel),0)
    @unittest.skipUnless(torch.cuda.is_available(),'CUDA hardware required')
    def test_cuda_coverage_agrees_with_cpu(self):
        scene=PointScene(frames=3,hidden=False);c=config();s,p,rgb,_=initialize_all_frame_zero(scene,c);results=[]
        for device in ('cpu','cuda'):results.append(SurfaceIndex(s,p,rgb,np.arange(len(s))).match(scene.observation(0,0),c,Backend(device)))
        assert_array_equal(results[0].point_id,results[1].point_id);assert_array_equal(results[0].explained,results[1].explained)
        assert_allclose(results[0].geometric_d2,results[1].geometric_d2,atol=1e-12)


if __name__=='__main__':unittest.main()
