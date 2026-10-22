import tempfile,unittest
import numpy as np
from numpy.testing import assert_allclose
from parallel_fixtures import Scene,configuration
from rgbd_common.backends.torch_cuda import Backend
from rgbd_common.fusion import fuse_frame_zero
from rgbd_common.lie import State
from rgbd_common.score import score_full_sequence,transactional_accept
from rgbd_common.score_cache import ScoreCache,table_energy
from rgbd_common.group_proposals import discover_rigid_groups


class OuterScoreTests(unittest.TestCase):
    def test_single_point_conditional_group_delta_vs_full_common_replay(self):
        from rgbd_common.lie import exp
        s=Scene(4,1);c=configuration();b=Backend('cpu');m,_=fuse_frame_zero(s,c,b);m.group[:]=1;m.group[10]=0
        states=[State(np.array([exp(np.array([.07*k,0,0,0,0,0])),np.eye(4)]),np.zeros((2,6))) for k in range(4)]
        with tempfile.TemporaryDirectory() as root:
            cache=ScoreCache(root+'/old',s,c,b);original=cache.build(m,states)
            proposed,ids,report=cache.propose_group_membership(m,states,1)
            self.assertEqual(ids.tolist(),[10]);complete=score_full_sequence(proposed,states,s,c,b)
            assert_allclose(complete['energy']-original['energy'],report['conditional_delta_sum'],atol=1e-10)
    def test_batched_backward_kernel_matches_scalar_queries_without_dropping_times(self):
        from rgbd_common.backward_cuda import backward_evidence
        from rgbd_common.fusion import backproject
        from rgbd_common.geometry import PointMap,RawObservationOwnership
        from rgbd_common.matching_cuda import match_rgbd
        from rgbd_common.backends.torch_cuda import numpy
        from scipy.special import log_ndtr
        s=Scene(4,2);c=configuration();b=Backend('cpu');m,_=fuse_frame_zero(s,c,b)
        s.observation(1,0).depth[3,3]=.7;s.observation(2,1).depth[1,1]=0;s.observation(0,0).depth[2,2]=.5
        states=[State(np.eye(4)[None],np.zeros((1,6))) for _ in s.times]
        xyz,rgb,cov,pix=backproject(s.observation(3,0),c,b);temporary=PointMap();temporary.append(xyz,rgb,np.zeros(len(xyz),int),cov,np.arange(len(xyz)))
        with tempfile.TemporaryDirectory() as root:
            cache=ScoreCache(root+'/score',s,c,b);cache.build(m,states);cache.refresh_ownership(m,RawObservationOwnership(s,root+'/ownership'))
            expected=np.zeros(len(xyz));strength=np.full(len(xyz),-np.inf);sf=np.full(len(xyz),-1);sc=sf.copy();sp=sf.copy();so=sf.copy()
            for k in (2,1,0):
                for ci in range(2):
                    frame=s.observation(k,ci);match,_,p,bins=match_rgbd(temporary,states[k],frame,c,b)
                    point=numpy(bins['visible_ids']);pixel=numpy(p['pixel'])[point];pixel=pixel[:,1]*frame.camera.width+pixel[:,0]
                    observed=frame.depth.ravel()[pixel];valid=np.isfinite(observed)&(observed>0)&frame.mask.ravel()[pixel];pred=numpy(p['depth'])[point][valid];point=point[valid];observed=observed[valid]
                    odds=log_ndtr((pred-observed)/np.sqrt(c.rgbd_covariance[3,3]+pred**2*c.rgbd_depth_squared[3,3]));odds[pred>observed]=0;expected[point]+=odds
                    mp=numpy(match.point);mu=numpy(match.pixel);gain=numpy(match.assignment_score-match.unmatched_score)
                    by_point=np.zeros(len(xyz));by_point[point]=odds;expected[mp]+=gain-by_point[mp]
                    use=gain>strength[mp];selected=mp[use];strength[selected]=gain[use];sf[selected]=k;sc[selected]=ci;sp[selected]=mu[use];so[selected]=cache.owners_at(k,ci,mu[use],m)
            actual=backward_evidence(xyz,rgb,0,states,s,[2,1,0],c,b,cache,m,batch_views=4)
            for x,y in zip(actual[:6],[expected,strength,sf,sc,sp,so]):assert_allclose(x,y,atol=1e-10)
            self.assertEqual(actual[6]['camera_group_queries'],6)
    def test_cached_full_and_incremental_scores_equal_complete_replay(self):
        s=Scene(3,1);c=configuration();b=Backend('cpu');m,_=fuse_frame_zero(s,c,b)
        states=[State(np.eye(4)[None],np.zeros((1,6))) for _ in s.times]
        with tempfile.TemporaryDirectory() as root:
            cache=ScoreCache(root+'/old',s,c,b);old=cache.build(m,states)
            full=score_full_sequence(m,states,s,c,b)
            assert_allclose(old['energy'],full['energy'],atol=1e-9)
            changed=m.copy();changed.xyz[10,2]+=.04;changed.rgb[10]=[.8,.1,.1]
            updated=cache.edit(changed,states,[10],root+'/candidate')
            complete=score_full_sequence(changed,states,s,c,b)
            assert_allclose(updated.score['energy'],complete['energy'],atol=1e-9)
            # Rejected edits need no disk factor copies. Acceptance must first
            # materialize every exact table before the parent can disappear.
            from pathlib import Path
            from rgbd_common.geometry import RawObservationOwnership
            self.assertEqual(list(Path(root+'/candidate').iterdir()),[])
            updated.refresh_ownership(changed,RawObservationOwnership(s,root+'/owners'))
            self.assertIsNone(updated.parent);self.assertIsNone(updated.proposal)
            for k in range(len(states)):
                self.assertAlmostEqual(updated.score['frames'][k]['energy'],table_energy(updated.read(k,0),changed.count),places=9)
            # Birth changes 1/N for EVERY existing association, even other rays.
            born=m.copy();born.append([[.05,.05,.7]],[[.1,.2,.8]],[0],m.covariance[:1],[100000])
            result=cache.edit(born,states,[m.count],root+'/birth')
            reference=score_full_sequence(born,states,s,c,b)
            assert_allclose(result.score['energy'],reference['energy'],atol=1e-9)
    def test_unchanged_estimate_cannot_be_an_edit_from_roundoff_only(self):
        from rgbd_common.outer_loop import same_estimate
        scene=Scene(3,1);config=configuration();backend=Backend('cpu');model,_=fuse_frame_zero(scene,config,backend)
        states=[State(np.eye(4)[None],np.zeros((1,6))) for _ in scene.times];copy=model.copy()
        copy.covariance*=.5
        self.assertTrue(same_estimate(model,copy,states,states))
        copy.xyz[0,0]+=1e-5
        self.assertFalse(same_estimate(model,copy,states,states))

    def test_rejected_transaction_does_not_commit(self):
        commits=[];records=[]
        accepted=transactional_accept({'energy':1.},{'energy':2.},lambda:commits.append(True),records.append)
        self.assertFalse(accepted);self.assertEqual(commits,[]);self.assertFalse(records[0]['accepted'])
    def test_multiframe_groups_discovered_without_static_category(self):
        rng=np.random.default_rng(21);source=rng.normal(size=(60,3))*.1
        targets=np.array([source.copy(),source.copy()]);targets[0,30:]+=[.15,.02,0];targets[1,30:]+=[.3,.04,0]
        covariance=np.broadcast_to(np.eye(3)*.0001,(2,60,3,3)).copy()
        transforms,weights,report=discover_rigid_groups(source,targets,covariance,configuration(),seed_budget=32)
        self.assertEqual(len(transforms),2);labels=weights.argmax(1)
        self.assertEqual(len(np.unique(labels[:30])),1);self.assertEqual(len(np.unique(labels[30:])),1);self.assertNotEqual(labels[0],labels[-1])
        # An identity transform is an ordinary estimated rigid component.
        self.assertLess(min(np.linalg.norm(t[1]-np.eye(4)) for t in transforms),1e-8)


class OuterPipelineTests(unittest.TestCase):
    def test_resume_preserves_initialization_and_replays_original_priors(self):
        from parallel_fixtures import DrawerScene
        from rgbd_common.outer_loop import run_outer
        from iekf.filter import IEKF
        s=DrawerScene();b=Backend('cpu');c=configuration();c.numerical['inner_iterations']=4
        with tempfile.TemporaryDirectory() as root:
            first=run_outer(IEKF(c,b),s,c,b,root,max_outer=1,visual=False)
            second=run_outer(IEKF(c,b),s,c,b,root,max_outer=1,visual=False,resume=True)
            self.assertEqual(first.initial_digest,second.initial_digest)
            self.assertTrue(any(row['stage']=='resume-accepted-checkpoint' for row in second.log.records))
            assert_allclose(second.steps[0].predicted_covariance,c.p0(len(second.states[0].poses)),atol=1e-14)
            self.assertTrue(any(row['stage']=='chronological-frontier-complete' and row['outer']==1 for row in second.log.records))
    def test_legacy_projection_rebase_requires_explicit_migration(self):
        import json
        from pathlib import Path
        from parallel_fixtures import DrawerScene
        from rgbd_common.outer_loop import run_outer
        from iekf.filter import IEKF
        scene=DrawerScene();backend=Backend('cpu');config=configuration();config.numerical['inner_iterations']=4
        with tempfile.TemporaryDirectory() as root:
            run_outer(IEKF(config,backend),scene,config,backend,root,max_outer=1,visual=False)
            path=Path(root)/'checkpoint.json';meta=json.loads(path.read_text());meta.pop('projection_operator_version');meta['energy']+=3.
            path.write_text(json.dumps(meta))
            with self.assertRaisesRegex(ValueError,'CHECKPOINT_NUMERICAL_OPERATOR_CHANGED'):
                run_outer(IEKF(config,backend),scene,config,backend,root,max_outer=1,visual=False,resume=True)
            result=run_outer(IEKF(config,backend),scene,config,backend,root,max_outer=1,visual=False,resume=True,rebase_legacy_projection=True)
            rebases=[r for r in result.log.records if r['stage']=='projection-score-rebase']
            self.assertEqual(len(rebases),1);self.assertFalse(rebases[0]['legacy_scores_validated'])
            self.assertAlmostEqual(rebases[0]['difference'],-3.)
            self.assertTrue((Path(root)/'reports/legacy_checkpoint_before_projection_v2.json').is_file())
            self.assertEqual(json.loads(path.read_text())['projection_operator_version'],2)

    def test_both_independent_full_forward_reverse_frontier_forward_cycles(self):
        from parallel_fixtures import DrawerScene
        from rgbd_common.outer_loop import run_outer
        from iekf.filter import IEKF
        from ukf.filter import UKF
        s=DrawerScene();b=Backend('cpu')
        for estimator_type in (IEKF,UKF):
            with self.subTest(pipeline=estimator_type.name),tempfile.TemporaryDirectory() as root:
                c=configuration();c.numerical['inner_iterations']=4
                result=run_outer(estimator_type(c,b),s,c,b,root,max_outer=1,visual=False)
                stages=[record['stage'] for record in result.log.records]
                self.assertGreaterEqual(stages.count('forward-complete'),2)
                self.assertIn('reverse-discovery-complete',stages);self.assertIn('chronological-frontier-complete',stages)
                self.assertTrue(all(row['accepted_energy']<=row['before_energy'] for row in result.history))
                self.assertGreater(max(np.linalg.norm(state.poses[:,:3,3],axis=1).max() for state in result.states),.05)
                self.assertGreaterEqual(len(result.states[0].poses),2)
                self.assertGreater(result.model.count,result.initial_points)
                result.model.source_owners([])
                from rgbd_common.outputs import export_result
                from pathlib import Path
                report=export_result(result,s,c,b,root,estimator_type.name,debug=True)
                self.assertTrue(report['complete_real_cycle']);self.assertEqual(report['source_pose_factors']['points'],result.model.count)
                self.assertTrue((Path(root)/'visual_debug/motion.html').is_file())
                self.assertFalse((Path(root)/'unresolved_observations.csv').is_symlink())
                self.assertFalse((Path(root)/'forward_depth').is_symlink())
                self.assertEqual((Path(root)/'unresolved_observations.csv').read_bytes(),(Path(root)/'unsolved_observations.csv').read_bytes())
                self.assertEqual(np.load(Path(root)/'group_poses.npz')['poses'].shape[0],len(s.times))
