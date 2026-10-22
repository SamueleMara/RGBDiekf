import tempfile,unittest
from pathlib import Path
import numpy as np
from numpy.testing import assert_allclose,assert_array_equal
import torch
from PIL import Image
from parallel_fixtures import Scene,configuration
from rgbd_common.backends.torch_cuda import Backend,numpy
from rgbd_common.foreground import exact_binary_mask,check_hard_mask_feasibility,foreground_depth_status,foreground_gray,VISIBLE,OCCLUDED,UNKNOWN
from rgbd_common.fusion import backproject,fuse_frame_zero
from rgbd_common.geometry import PointMap,ObservationIDs,RawObservationOwnership
from rgbd_common.lie import State
from rgbd_common.matching_cuda import match_rgbd
from rgbd_common.score import frame_score,score_full_sequence
from rgbd_common.score_cache import ScoreCache


def singleton(scene):
    frame=scene.observation(0,0);point=PointMap()
    point.append([[.05,.05,1.]],[[.2,.3,.4]],[0],np.eye(6)[None]*.001,[36])
    return point,State(np.eye(4)[None],np.zeros((1,6)))


class HardForegroundTests(unittest.TestCase):
    def test_exact_binary_values_and_invalid_greys(self):
        for values in (np.array([[0,1]]),np.array([[0,255]]),np.array([[False,True]])):
            assert_array_equal(exact_binary_mask(values),[[False,True]])
        for bad in (np.array([[0,128,255]]),np.array([[0,1,255]]),np.array([[0.,.5]]),np.zeros((2,2,3))):
            with self.assertRaisesRegex(ValueError,'binary|BINARY'):exact_binary_mask(bad)

    def test_missing_misaligned_and_nonbinary_dataset_masks_fail(self):
        from data_fixtures import make_dataset
        from rgbd_common.dataloader import load_dataset
        with tempfile.TemporaryDirectory() as root:
            path=Path(root);make_dataset(path);(path/'metadata/config.yaml').write_text('fps: 25\n')
            mask=path/'masks/cam_000/frame_000000.png';data=load_dataset(path)
            Image.fromarray(np.full((2,3),255,np.uint8)).save(mask)
            with self.assertRaisesRegex(ValueError,'shape mismatch'):data.observation(0,0)
            Image.fromarray(np.full((2,2),128,np.uint8)).save(mask)
            with self.assertRaisesRegex(ValueError,'NON_BINARY'):data.observation(0,0)
            for m in path.glob('masks/*/*.png'):m.unlink()
            with self.assertRaisesRegex(ValueError,'MISSING_FOREGROUND'):load_dataset(path).observation(0,0)

    def test_frame_zero_backprojection_and_unsolved_source_domain(self):
        scene=Scene(3,2);config=configuration();backend=Backend('cpu')
        for row in scene.images:
            for frame in row:frame.mask[:,:2]=False;frame.depth[0,2]=0
        model,_=fuse_frame_zero(scene,config,backend);raw=ObservationIDs(scene)
        for k,c,pixel in zip(*raw.decode(model.provenance.raw_ids)):self.assertTrue(scene.observation(k,c).valid.ravel()[pixel])
        with self.assertRaisesRegex(ValueError,'foreground'):backproject(scene.observation(0,0),config,backend,[0])
        with tempfile.TemporaryDirectory() as root:
            own=RawObservationOwnership(scene,root)
            for k in range(3):
                for ci in range(2):
                    frame=scene.observation(k,ci);matches,*_=match_rgbd(model,State(np.eye(4)[None],np.zeros((1,6))),frame,config,backend)
                    own.write(k,ci,numpy(matches.pixel),model.point_id[numpy(matches.point)])
                    remaining=own.unsolved(k,ci,frame)
                    self.assertTrue(frame.valid.ravel()[remaining].all());self.assertEqual(int(frame.valid.sum()),len(matches)+len(remaining))

    def test_background_sources_ownership_and_forged_filter_targets_are_rejected(self):
        from rgbd_common.measurement_problem import MeasurementProblem
        from rgbd_common.group_proposals import world_at
        scene=Scene(3,1);config=configuration();backend=Backend('cpu');model,state=singleton(scene);frame=scene.observation(1,0)
        frame.mask[0,0]=False;frame.depth[0,0]=.5
        with self.assertRaisesRegex(ValueError,'in-image'):backproject(frame,config,backend,[-1])
        position,valid=world_at(frame,np.array([[0.,0.],[4.,4.]]))
        self.assertFalse(valid[0]);self.assertTrue(np.isnan(position[0]).all());self.assertTrue(valid[1])
        with tempfile.TemporaryDirectory() as root:
            ownership=RawObservationOwnership(scene,root)
            with self.assertRaisesRegex(ValueError,'Background'):ownership.write(1,0,[0],[0])
        match,*_=match_rgbd(model,state,frame,config,backend);self.assertEqual(len(match),1);match.pixel[0]=0
        with self.assertRaisesRegex(ValueError,'foreground'):MeasurementProblem(model,[match],backend)

    def test_negative_support_uses_only_declared_premise_and_never_background_depth(self):
        scene=Scene(3,1);config=configuration();backend=Backend('cpu');model,state=singleton(scene);frame=scene.observation(0,0)
        frame.mask[:]=False
        for depth in (2.,.5,0.,np.nan,np.inf,-10.):
            frame.depth[:]=depth
            result=check_hard_mask_feasibility(model,state,frame,config,backend)
            self.assertFalse(result['feasible']);self.assertEqual(numpy(result['status']).tolist(),[VISIBLE])
            self.assertEqual(len(match_rgbd(model,state,frame,config,backend)[0]),0)
        for premise in (None,False):
            frame.object_only_no_external_occluders=premise
            result=check_hard_mask_feasibility(model,state,frame,config,backend)
            self.assertTrue(result['feasible']);self.assertEqual(numpy(result['status']).tolist(),[UNKNOWN])

    def test_foreground_hiding_uses_likelihood_assignment_and_invalid_depth_is_unknown(self):
        backend=Backend('cpu');z=backend.tensor([1.,1.,1.,1.]);d=backend.tensor([.999,.5,2.,0.])
        status=foreground_depth_status(z,d,backend.tensor([True,False,False,False],torch.bool))
        assert_array_equal(numpy(status),[VISIBLE,OCCLUDED,VISIBLE,UNKNOWN])
        scene=Scene(3,1);config=configuration();model,state=singleton(scene);frame=scene.observation(0,0)
        frame.depth[:]=.5
        result=check_hard_mask_feasibility(model,state,frame,config,backend)
        self.assertTrue(result['feasible']);self.assertEqual(numpy(result['status']).tolist(),[OCCLUDED]);self.assertEqual(model.count,1)
        self.assertEqual(len(match_rgbd(model,state,frame,config,backend)[0]),0)
        frame.depth[:]=0
        self.assertEqual(numpy(check_hard_mask_feasibility(model,state,frame,config,backend)['status']).tolist(),[UNKNOWN])

    def test_background_rgb_cannot_change_score_matches_or_flow_image(self):
        scene=Scene(3,1);config=configuration();backend=Backend('cpu');model,state=singleton(scene);frame=scene.observation(0,0)
        frame.mask[:,:2]=False;frame.depth[:,:2]=.1
        before=frame_score(model,state,scene,0,config,backend);gray=foreground_gray(frame)
        frame.rgb[:,:2]=[1.,0.,1.]
        after=frame_score(model,state,scene,0,config,backend)
        self.assertEqual(before[0],after[0]);assert_array_equal(numpy(before[1][0].pixel),numpy(after[1][0].pixel));assert_array_equal(gray,foreground_gray(frame))

    def test_changed_map_cache_checks_background_constraints_all_times(self):
        scene=Scene(3,1);config=configuration();backend=Backend('cpu');model,state=singleton(scene);states=[state]*3
        scene.observation(1,0).depth[4,4]=.5
        scene.observation(1,0).mask[6,6]=False;scene.observation(1,0).depth[6,6]=2
        with tempfile.TemporaryDirectory() as root:
            cache=ScoreCache(root+'/old',scene,config,backend);cache.build(model,states)
            # Background cannot be a match; an initially hidden point is valid.
            self.assertTrue(np.isfinite(cache.score['energy']))
            changed=model.copy();changed.xyz[0,2]=.25
            # Same projection moves to (6,6); place explicit measured background
            # there so this candidate is visible and violates the hard mask.
            candidate=cache.edit(changed,states,[0],root+'/candidate')
            complete=score_full_sequence(changed,states,scene,config,backend)
            self.assertTrue(np.isinf(candidate.score['energy']));self.assertTrue(np.isinf(complete['energy']))
            self.assertGreater(candidate.score['mask_constraint_violations'],0)

    def test_real_backward_discovery_keeps_hidden_late_point_and_rejects_visible_background(self):
        from rgbd_common.discovery import reverse_match_unsolved
        scene=Scene(3,1);config=configuration();backend=Backend('cpu');model,state=singleton(scene);states=[state]*3
        scene.observation(0,0).depth[2,2]=.5
        with tempfile.TemporaryDirectory() as root:
            cache=ScoreCache(root+'/score',scene,config,backend);cache.build(model,states)
            cache.refresh_ownership(model,RawObservationOwnership(scene,root+'/owners'))
            proposal,report=reverse_match_unsolved(model,states,scene,2,0,[18],config,backend,cache,target_frames=[1,0])
            self.assertIsNotNone(proposal);new_model,_=proposal
            self.assertEqual(new_model.count,2);self.assertEqual(report['target_times'],2)
            self.assertTrue(np.isfinite(score_full_sequence(new_model,states,scene,config,backend)['energy']))
            scene.observation(0,0).mask[2,2]=False
            self.assertTrue(np.isinf(score_full_sequence(new_model,states,scene,config,backend)['energy']))

    def test_each_camera_and_nearest_pixel_boundary_are_independent(self):
        from rgbd_common.projection_cuda import transport_and_project
        scene=Scene(3,2);config=configuration();backend=Backend('cpu');model,state=singleton(scene)
        scene.observation(0,0).mask[:]=False;scene.observation(0,0).depth[:]=2
        self.assertFalse(check_hard_mask_feasibility(model,state,scene.observation(0,0),config,backend)['feasible'])
        self.assertTrue(check_hard_mask_feasibility(model,state,scene.observation(0,1),config,backend)['feasible'])
        points=backend.tensor([[-.00001,0,1],[.00001,0,1]],backend.input_dtype)
        projection=transport_and_project(points,backend.tensor([0,0],torch.long),backend.tensor(state.poses,backend.input_dtype),scene.cameras[0])
        assert_array_equal(numpy(projection['pixel'])[:,0],[3,4])

    def test_both_filter_candidates_respect_the_same_feasible_set(self):
        from iekf.filter import IEKF
        from ukf.filter import UKF
        scene=Scene(3,1);config=configuration();backend=Backend('cpu');model,base=singleton(scene)
        frame=scene.observation(1,0);frame.mask[:,5:]=False;frame.depth[:,5:]=2
        desired=np.zeros(12);desired[0]=.2
        class ConstrainedProblem:
            def feasible(self,state):return check_hard_mask_feasibility(model,state,frame,config,backend)['feasible']
            def evaluate(self,state):return .5*1000*np.sum((state.minus(base)-desired)**2) if self.feasible(state) else float('inf')
        problem=ConstrainedProblem();prior=np.eye(12)*.01
        def analytical(problem,state):return backend.tensor(np.eye(12)[None]*1000),backend.tensor((1000*(desired-state.minus(base)))[None])
        def sigma(problem,state,covariance,parameters):
            a=np.zeros((12,25));a[:,:12]=np.eye(12)*.1;m=np.eye(25)*2;b=np.zeros(25);b[0]=4
            return a,backend.tensor(m),backend.tensor(b),{}
        first=IEKF(config,backend,analytical).correct(base,prior,problem)
        second=UKF(config,backend,sigma).correct(base,prior,problem)
        self.assertTrue(problem.feasible(first[0]));self.assertTrue(problem.feasible(second[0]))
        assert_allclose(second[0].poses,base.poses);assert_allclose(second[1],prior)
        self.assertTrue(second[2]['unresolved_mask_visibility_conflict'])
        self.assertGreater(first[0].poses[0,0,3],0)

    def test_mask_diagnostic_exports_are_actual_pixel_sets(self):
        from rgbd_common.outputs import export_foreground_masks
        scene=Scene(3,1);config=configuration();backend=Backend('cpu');model,state=singleton(scene)
        frame=scene.observation(0,0);frame.mask[:,:2]=False;frame.depth[:,:2]=.1
        matches,_,projected,bins=match_rgbd(model,state,frame,config,backend)
        with tempfile.TemporaryDirectory() as root:
            report=export_foreground_masks(root,model,state,frame,matches,projected,bins,config,backend)
            self.assertEqual(report['mask_constraint_violations'],0)
            for name in ('fg_mask_observed','projected_point_valid','projected_visible_fg','matched_fg','unmatched_fg'):
                self.assertTrue((Path(root)/name/'frame_000000_cam_000.png').is_file())
            with Image.open(Path(root)/'fg_mask_observed/frame_000000_cam_000.png') as image:assert_array_equal(np.asarray(image)>0,frame.mask)
            with Image.open(Path(root)/'matched_fg/frame_000000_cam_000.png') as image:owned=np.asarray(image)>0
            with Image.open(Path(root)/'unmatched_fg/frame_000000_cam_000.png') as image:unmatched=np.asarray(image)>0
            assert_array_equal(owned|unmatched,frame.valid);self.assertFalse((owned&unmatched).any())
            self.assertEqual((Path(root)/'mask_constraint_violations.csv').read_text().splitlines(),['frame,camera,point_id,pose_stage,visibility'])


@unittest.skipUnless(torch.cuda.is_available(),'Actual host CUDA is required')
class HardForegroundGPUTests(unittest.TestCase):
    def test_backward_kernel_rejects_visible_background_without_reading_its_rgb(self):
        from types import SimpleNamespace
        from rgbd_common.backward_cuda import backward_evidence
        scene=Scene(3,1);config=configuration();model,state=singleton(scene);frame=scene.observation(0,0)
        frame.mask[4,4]=False;frame.depth[4,4]=2
        owner=SimpleNamespace(owners_at=lambda k,c,pixels,model:np.full(len(pixels),-1,np.int64))
        for device in ('cpu','cuda'):
            backend=Backend(device)
            result=backward_evidence(model.xyz,model.rgb,0,[state]*3,scene,[0],config,backend,owner,model)
            self.assertTrue(np.isneginf(result[0][0]));self.assertEqual(result[6]['hard_mask_infeasible_point_hypotheses'],1)
            frame.object_only_no_external_occluders=None;frame.depth[4,4]=.5
            hidden=backward_evidence(model.xyz,model.rgb,0,[state]*3,scene,[0],config,backend,owner,model)
            self.assertEqual(hidden[0][0],0);self.assertEqual(hidden[6]['hard_mask_infeasible_point_hypotheses'],0)
            frame.depth[4,4]=2;frame.object_only_no_external_occluders=True

    def test_foreground_gpu_backprojection_and_full_noise_pushforward(self):
        from rgbd_common.lie import exp
        from rgbd_common.observation_model import covariance
        scene=Scene(3,1);config=configuration();frame=scene.observation(0,0)
        frame.camera.world_to_camera=exp(np.array([.1,-.2,.3,.2,.1,-.15]))
        frame.mask[::2,::2]=False;frame.depth[:]=np.linspace(.8,1.2,64).reshape(8,8)
        frame.depth[0,1]=np.nan;frame.depth[1,2]=np.inf;frame.depth[2,3]=0
        root=np.array([[.04,0,0,0],[.01,.05,0,0],[0,.01,.06,0],[.001,0,.002,.01]])
        config.rgbd_covariance=root@root.T
        ids=np.flatnonzero(frame.valid.ravel());y,x=np.divmod(ids,8);z=frame.depth.ravel()[ids]
        fx,fy,cx,cy=frame.camera.intrinsics;ray=np.c_[(x-cx)/fx,(y-cy)/fy,np.ones(len(ids))]
        cw=frame.camera.world_to_camera;r=cw[:3,:3];expected=(ray*z[:,None]-cw[:3,3])@r
        jac=np.zeros((len(ids),6,6));jac[:,:3,0]=r.T[:,0]*z[:,None]/fx;jac[:,:3,1]=r.T[:,1]*z[:,None]/fy;jac[:,:3,2]=ray@r;jac[:,3:,3:]=np.eye(3)
        cpu=Backend('cpu');noise=numpy(covariance(config,cpu.tensor(z),cpu));reference=jac@noise@jac.transpose(0,2,1)
        for device in ('cpu','cuda'):
            actual=backproject(frame,config,Backend(device))
            assert_array_equal(actual[3],ids);assert_allclose(actual[0],expected,atol=1e-14);assert_allclose(actual[2],reference,atol=1e-14)
            assert_array_equal(actual[1],frame.rgb.reshape(-1,3)[ids])
            with self.assertRaisesRegex(ValueError,'foreground'):backproject(frame,config,Backend(device),[0])

    def test_actual_backward_evidence_gpu_cpu_masks_occlusion_and_raw_support(self):
        from rgbd_common.backward_cuda import backward_evidence
        scene=Scene(4,2);config=configuration();cpu=Backend('cpu');model,_=fuse_frame_zero(scene,config,cpu)
        states=[State(np.eye(4)[None],np.zeros((1,6))) for _ in scene.times]
        scene.observation(0,0).mask[:,:2]=False;scene.observation(0,0).depth[:,:2]=.5
        scene.observation(1,1).depth[3,3]=.975
        scene.observation(2,0).depth[1,1]=0
        xyz,rgb,_,_=backproject(scene.observation(3,0),config,cpu);results=[]
        with tempfile.TemporaryDirectory() as root:
            for device in ('cpu','cuda'):
                backend=Backend(device);cache=ScoreCache(root+'/'+device,scene,config,backend);cache.build(model,states)
                cache.refresh_ownership(model,RawObservationOwnership(scene,root+'/'+device+'_owners'))
                results.append(backward_evidence(xyz,rgb,0,states,scene,[2,1,0],config,backend,cache,model,batch_views=4))
        for first,second in zip(results[0][:6],results[1][:6]):assert_allclose(first,second,rtol=1e-10,atol=1e-10)
        self.assertEqual(results[0][6]['camera_group_queries'],6);self.assertEqual(results[1][6]['camera_group_queries'],6)

    def test_gpu_cpu_visibility_matching_and_information_agree(self):
        from rgbd_common.measurement_problem import MeasurementProblem
        from iekf.measurement import reduce_information
        from ukf.measurement import measurement_moments
        scene=Scene(3,2);config=configuration();model,state=singleton(scene)
        frame=scene.observation(1,0);frame.mask[:,:2]=False;frame.depth[:,:2]=.5
        result=[]
        for device in ('cpu','cuda'):
            backend=Backend(device);match,*_=match_rgbd(model,state,frame,config,backend)
            status=check_hard_mask_feasibility(model,state,frame,config,backend)
            problem=MeasurementProblem(model,[match],backend);info,rhs=reduce_information(problem,state);a,m,b,_=measurement_moments(problem,state,np.eye(12)*1e-7)
            result.append([numpy(status['status']),numpy(match.pixel),numpy(info),numpy(rhs),a,numpy(m),numpy(b)])
        for first,second in zip(*result):assert_allclose(first,second,rtol=1e-5,atol=1e-5)
