"""V2 M1--M8 branch-specific invariance and discrete support regressions."""
import json,os,tempfile,unittest
from pathlib import Path
import numpy as np
import torch
from numpy.testing import assert_allclose,assert_array_equal
from parallel_fixtures import Scene,configuration
from test_hard_foreground import singleton
from rgbd_common.backends.torch_cuda import Backend,numpy
from rgbd_common.foreground import compact_valid_fg,check_hard_mask_feasibility
from rgbd_common.fusion import backproject,fuse_frame_zero
from rgbd_common.matching_cuda import match_rgbd
from rgbd_common.measurement_problem import MeasurementProblem
from rgbd_common.score import score_full_sequence
from rgbd_common.score_cache import ScoreCache
from rgbd_common.geometry import RawObservationOwnership
from rgbd_common.discovery import reverse_match_unsolved
from iekf.filter import IEKF
from ukf.filter import UKF


def record(case,branch,device,raw_ids,assertions):
    root=os.environ.get('RGBD_TEST_REPORTS')
    if root:
        directory=Path(root)/'mask_port_cases';directory.mkdir(exist_ok=True)
        (directory/(case+'.json')).write_text(json.dumps({'test':case,'branch':branch,'device':device,'raw_source_ids':list(map(int,raw_ids)),'assertions':assertions,'status':'PASS'},indent=2))


class MaskedV2Tests(unittest.TestCase):
    def preparation(self,estimator_type,device):
        backend=Backend(device);scene=Scene(3,1);config=configuration();frame=scene.observation(0,0)
        frame.mask[:2]=False;frame.raw_depth_validity=np.ones((8,8),bool);frame.raw_depth_validity[2,:2]=False
        frame.depth[3,0]=np.nan;frame.depth[3,1]=0
        assert_array_equal(frame.prepared_rgb,np.asarray(frame.mask)[:,:,None]*frame.rgb)
        expected=frame.mask&frame.raw_depth_validity&np.isfinite(frame.depth)&(frame.depth>0)
        assert_array_equal(frame.valid,expected)
        ids=compact_valid_fg(backend.tensor(frame.mask,torch.bool),backend.tensor(frame.raw_depth_validity,torch.bool),backend.tensor(frame.depth))
        assert_array_equal(numpy(ids),np.flatnonzero(expected.ravel()));actual=backproject(frame,config,backend)
        assert_array_equal(actual[3],numpy(ids))
        record(self._testMethodName,estimator_type.name,device,actual[3],['P=M*I pixelwise; reference_3dgs_masked.pdf Eq41','Vused=M AND Vraw AND finite positive depth','GPU compact IDs equal independent NumPy; no invalid source'])

    def invariance(self,estimator_type,device):
        backend=Backend(device);scene=Scene(3,1);config=configuration();model,state=singleton(scene)
        for row in scene.images:
            frame=row[0];frame.mask[:,:2]=False
            frame.raw_depth_validity=np.ones((8,8),bool);frame.raw_depth_validity[0,2:]=False
        prior=np.eye(12)*1e-7
        def evaluate(root):
            frame=scene.observation(1,0);match,*_=match_rgbd(model,state,frame,config,backend)
            problem=MeasurementProblem(model,[match],backend)
            result=estimator_type(config,backend).correct(state,prior,problem)
            cache=ScoreCache(root+'/score',scene,config,backend);cache.build(model,[state]*3)
            cache.refresh_ownership(model,RawObservationOwnership(scene,root+'/owner'))
            proposal,details=reverse_match_unsolved(model,[state]*3,scene,2,0,[18],config,backend,cache,target_frames=[1,0])
            points=None if proposal is None else proposal[0]
            init,_=fuse_frame_zero(scene,config,backend)
            return [numpy(match.pixel),numpy(match.observed),result[0].poses,result[0].twists,result[1],np.array(score_full_sequence(model,[state]*3,scene,config,backend)['energy']),init.xyz,init.rgb,init.source_id,
                    np.empty((0,3)) if points is None else points.xyz,np.empty(0,dtype=np.int64) if points is None else points.provenance.raw_ids]
        with tempfile.TemporaryDirectory() as root:
            baseline=evaluate(root+'/initial');rng=np.random.default_rng(801)
            for index,value in enumerate((0.,np.nan,np.inf,1e20,-50.)):
                for row in scene.images:
                    frame=row[0];background=~frame.mask;invalid=~frame.raw_depth_validity
                    frame.rgb[background]=rng.uniform(0,1,(int(background.sum()),3))
                    frame.depth[background|invalid]=value
                changed=evaluate(root+f'/trial_{index}')
                for expected,actual in zip(baseline,changed):assert_array_equal(actual,expected)
        record(self._testMethodName,estimator_type.name,device,model.source_id,['Arbitrary background RGB/depth and explicitly invalid FG depth give bit-identical targets, pose, twist, covariance, full J, fused initialization and backward-discovered provenance'])

    def empty(self,estimator_type,device):
        scene=Scene(3,1);backend=Backend(device);config=configuration();model,state=singleton(scene)
        for row in scene.images:row[0].mask[:]=False
        self.assertEqual(len(backproject(scene.observation(0,0),config,backend)[3]),0)
        with self.assertRaisesRegex(ValueError,'NO_FRAME_ZERO_GEOMETRY'):fuse_frame_zero(scene,config,backend)
        match,*_=match_rgbd(model,state,scene.observation(1,0),config,backend)
        self.assertEqual(len(match),0);problem=MeasurementProblem(model,[match],backend)
        result=estimator_type(config,backend).correct(state,np.eye(12)*1e-5,problem)
        assert_array_equal(result[0].poses,state.poses)
        with tempfile.TemporaryDirectory() as root:
            cache=ScoreCache(root,scene,config,backend);cache.build(model,[state]*3)
            proposal,_=reverse_match_unsolved(model,[state]*3,scene,2,0,[],config,backend,cache)
            self.assertIsNone(proposal)
        record(self._testMethodName,estimator_type.name,device,[],['All-zero masks: no initialization, match factor, motion correction, or backward insertion'])

    def once_visible(self,estimator_type,device):
        scene=Scene(3,1);config=configuration();backend=Backend(device);model,state=singleton(scene)
        for k in (0,1):scene.observation(k,0).depth[2,2]=0
        with tempfile.TemporaryDirectory() as root:
            cache=ScoreCache(root+'/score',scene,config,backend);cache.build(model,[state]*3)
            cache.refresh_ownership(model,RawObservationOwnership(scene,root+'/owner'))
            proposal,details=reverse_match_unsolved(model,[state]*3,scene,2,0,[18],config,backend,cache)
            self.assertIsNone(proposal);self.assertEqual(details['supported'],0)
        record(self._testMethodName,estimator_type.name,device,[2*64+18],['Once-only-visible source with no earlier evidence remains unresolved; no forced group or insertion'])

    def test_explicit_raw_validity_and_acquisition_premise_directory_zip_and_cache(self):
        import zipfile
        from PIL import Image
        from test_data_loading import real_scene
        from rgbd_common.dataloader import load_dataset
        from rgbd_common.frame_cache import CachedDataset
        with tempfile.TemporaryDirectory() as root:
            source=Path(root)/'scene';source.mkdir();real_scene(source)
            (source/'metadata/observation_model.json').write_text(json.dumps({'object_only_no_external_occluders':True}))
            for path in (source/'depth_npy').rglob('*.npy'):
                target=source/'depth_validity'/path.relative_to(source/'depth_npy');target.parent.mkdir(parents=True,exist_ok=True)
                np.save(target,np.array([[0,1],[1,1]],np.uint8))
            data=load_dataset(source);original=data.observation(0,0)
            self.assertFalse(original.raw_valid[0,0]);self.assertTrue(original.object_only_no_external_occluders)
            cached=CachedDataset(data,Path(root)/'cache').observation(0,0);assert_array_equal(cached.valid,original.valid)
            with zipfile.ZipFile(Path(root)/'scene.zip','w') as archive:
                for path in source.rglob('*'):
                    if path.is_file():archive.write(path,path.relative_to(source))
            zipped=load_dataset(Path(root)/'scene.zip').observation(0,0)
            assert_array_equal(zipped.raw_valid,original.raw_valid);self.assertTrue(zipped.object_only_no_external_occluders)
            self.assertIn('raw_validity_sha256',data.observation_fingerprints['000000_000'])

    def test_immutable_point_id_tie_order_and_sparse_buffers(self):
        from rgbd_common.projection_cuda import transport_and_project,build_pixel_bins_and_zbuffer
        scene=Scene();backend=Backend('cpu');model,state=singleton(scene)
        xyz=backend.tensor(np.repeat(model.xyz,3,0),backend.input_dtype)
        projection=transport_and_project(xyz,backend.tensor([0,0,0],torch.long),backend.tensor(state.poses,backend.input_dtype),scene.cameras[0])
        bins=build_pixel_bins_and_zbuffer(projection,scene.cameras[0],backend.tensor([90,3,7],torch.long))
        assert_array_equal(numpy(bins['visible_ids']),[1]);self.assertEqual(tuple(bins['first_index'].shape),(8,8));self.assertEqual(tuple(bins['first_depth'].shape),(8,8))

    def test_reordered_point_ids_preserve_full_and_incremental_score_ties(self):
        from rgbd_common.geometry import PointMap
        scene=Scene(3,1);backend=Backend('cpu');config=configuration();_,state=singleton(scene)
        model=PointMap();rgb=scene.observation(0,0).rgb[4,4]
        model.append([[.05,.05,1.],[.05,.05,1.]],[[1.,0.,0.],rgb],[0,0],np.tile(np.eye(6)*.001,(2,1,1)),[36,37])
        model=model.take([1,0]);states=[state]*3
        with tempfile.TemporaryDirectory() as root:
            cache=ScoreCache(root+'/base',scene,config,backend);cache.build(model,states)
            reference=score_full_sequence(model,states,scene,config,backend)
            assert_allclose(cache.score['energy'],reference['energy'],atol=1e-10)
            changed=model.copy();changed.rgb[0]=[.1,.1,.1]
            trial=cache.edit(changed,states,[0],root+'/trial')
            assert_allclose(trial.score['energy'],score_full_sequence(changed,states,scene,config,backend)['energy'],atol=1e-10)
            trial.refresh_ownership(changed,RawObservationOwnership(scene,root+'/owners'))
            assert_array_equal(trial.read(1,0)['point'],[1,0])

    def test_fast_feasibility_is_identical_to_full_status_without_reading_bg_depth(self):
        scene=Scene(3,1);backend=Backend('cpu');config=configuration();model,state=singleton(scene);frame=scene.observation(1,0)
        for premise in (True,False,None):
            frame.object_only_no_external_occluders=premise
            for foreground in (True,False):
                frame.mask[:]=foreground
                for depth in (0.,.5,2.,np.nan):
                    frame.depth[:]=depth
                    full=check_hard_mask_feasibility(model,state,frame,config,backend)
                    fast=check_hard_mask_feasibility(model,state,frame,config,backend,classify_foreground=False)
                    self.assertEqual(full['feasible'],fast['feasible']);assert_array_equal(numpy(full['violation_indices']),numpy(fast['violation_indices']))

    def test_cached_known_depth_backward_evidence_equals_full_projector_and_rolls_back(self):
        from types import SimpleNamespace
        from rgbd_common.backward_cuda import backward_evidence
        scene=Scene(3,1);backend=Backend('cpu');config=configuration();model,state=singleton(scene);states=[state]*3
        xyz,rgb,_,_=backproject(scene.observation(2,0),config,backend)
        with tempfile.TemporaryDirectory() as root:
            cache=ScoreCache(root+'/base',scene,config,backend);cache.build(model,states)
            cache.refresh_ownership(model,RawObservationOwnership(scene,root+'/owners'))
            cached=backward_evidence(xyz,rgb,0,states,scene,[1,0],config,backend,cache,model)
            reference=backward_evidence(xyz,rgb,0,states,scene,[1,0],config,backend,SimpleNamespace(owners_at=cache.owners_at),model)
            for a,b in zip(cached[:6],reference[:6]):assert_allclose(a,b,rtol=0,atol=0)
            before=cache.front_depth_at(1,0,[36]).copy();trial=model.copy();trial.xyz[0,2]=.5
            candidate=cache.edit(trial,states,[0],root+'/trial');candidate.discard()
            assert_array_equal(cache.front_depth_at(1,0,[36]),before)

    def test_sigma_quadrature_keeps_fixed_raw_factors_across_raster_changes(self):
        from ukf.measurement import measurement_moments
        scene=Scene(3,1);config=configuration();backend=Backend('cpu');model,state=singleton(scene)
        matches=[match_rgbd(model,state,scene.observation(1,0),config,backend)[0]]
        problem=MeasurementProblem(model,matches,backend)
        _,_,_,small=measurement_moments(problem,state,np.eye(12)*1e-9)
        _,m,b,large=measurement_moments(problem,state,np.eye(12)*.01)
        self.assertEqual(small['active_factors'],1);self.assertEqual(large['active_factors'],1)
        self.assertGreater(np.linalg.norm(numpy(m)-np.eye(25)),1.)
        self.assertEqual(large['deferred_visibility_factors'],0)
        # Another group can enter the target's bin at one sigma even though
        # it is outside that bin at the nominal state.
        from rgbd_common.lie import State
        model.append([[.15,.025,.5]],[[.2,.3,.4]],[1],np.eye(6)[None]*.001,[99])
        state=State(np.tile(np.eye(4),(2,1,1)),np.zeros((2,6)))
        match=match_rgbd(model,state,scene.observation(1,0),config,backend)[0]
        p=np.eye(24)*1e-9;p[12,12]=(.125/np.sqrt(24))**2
        _,_,_,report=measurement_moments(MeasurementProblem(model,[match],backend),state,p)
        self.assertEqual(report['active_factors'],len(match))
        self.assertEqual(report['deferred_visibility_factors'],0)

    def test_fixed_target_depth_noise_is_not_an_occluding_second_observation(self):
        from ukf.measurement import measurement_moments,project_sigma,woodbury
        from ukf.sigma_points import sigma_points
        scene=Scene(3,1);config=configuration();backend=Backend('cpu');model,state=singleton(scene)
        model.xyz[0]=[0,0,1.]
        match=match_rgbd(model,state,scene.observation(1,0),config,backend)[0]
        self.assertEqual(len(match),1)
        prior=np.eye(12)*1e-12;prior[2,2]=.01
        problem=MeasurementProblem(model,[match],backend)
        a,m,b,report=measurement_moments(problem,state,prior)
        self.assertEqual(report['active_factors'],1)
        eta,wm,wc=sigma_points(prior);states=[state.plus(e) for e in eta]
        h,_=project_sigma(backend.tensor(np.array([s.poses for s in states])),problem.xyz,problem.rgb,problem.group,match.camera,backend)
        values=numpy(h[:,0]);mean=wm@values;z=(values-mean).T*np.sqrt(wc)[None]
        residual=numpy(match.observed[0])-mean;r=numpy(match.covariance[0]);innovation=r+z@z.T;cross=a@z.T
        delta,cov=woodbury(a,prior,numpy(m),numpy(b))
        assert_allclose(delta,cross@np.linalg.solve(innovation,residual),atol=1e-12)
        assert_allclose(cov,prior-cross@np.linalg.solve(innovation,cross.T),atol=1e-12)
        self.assertLess(cov[2,2],prior[2,2])


for estimator in (IEKF,UKF):
    for device in ('cpu','cuda'):
        for operation in ('preparation','invariance','empty','once_visible'):
            def method(self,e=estimator,d=device,op=operation):return getattr(self,op)(e,d)
            if device=='cuda':method=unittest.skipUnless(torch.cuda.is_available(),'Actual host CUDA is required')(method)
            setattr(MaskedV2Tests,f'test_{operation}_{estimator.name.lower()}_{device}',method)
