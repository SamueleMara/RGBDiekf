"""Final pair gates. CPU fixtures and narrowly scoped real frame-1 checks."""
import ast
import copy
import hashlib
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
from scipy.linalg import block_diag
from tools import f1_joint_responsibility_audit as J
from tools import root_cause_oracle as O
from tests.test_f1_temporal_reference_diagnostics import cam,textured,noise
from tests.projective_fixtures import POLICY

ROOT=Path(os.environ.get('F1_JOINT_FORENSICS_ROOT',str(J.DEFAULT_ROOT)))


def simple_row(c,x,appearance,im,pair_index):
    row=O.row(im,c,x,appearance,noise(),POLICY,np.eye(3)*1e-4,np.zeros(3));row.update(camera=0,pair_index=pair_index)
    return row


class JointMathTests(unittest.TestCase):
    def test_disjoint_assignment_has_zero_cross_information(self):
        rows=[{'pair_index':j,'r':np.array([.01,.02,.03,.001]),'H':np.r_[np.eye(3),[[0,0,1]]]} for j in range(2)]
        info=J.joint_information(np.zeros(12),np.zeros(12),np.eye(12),rows,np.eye(8))
        np.testing.assert_array_equal(info['A'][:6,6:],0);np.testing.assert_array_equal(info['covariance'][:6,6:],0)
        np.testing.assert_allclose(info['A']@info['delta'],info['b'],atol=1e-14)

    def test_shared_bilinear_raw_pixel_covariance_matches_linear_map(self):
        c=cam();im=textured();x=np.array([.013,-.007,1.55]);a=np.array([.5,.5,.5])
        r1=simple_row(c,x,a,im,0);r2=simple_row(c,x+np.array([.001,0.,0.]),a,im,1);R=J.shared_sensor_covariance([r1,r2],[im],noise(),POLICY)
        np.testing.assert_allclose(np.diag(R)[:4],r1['var'],rtol=1e-12);np.testing.assert_allclose(np.diag(R)[4:],r2['var'],rtol=1e-12)
        supports=[J.footprint(r) for r in (r1,r2)];pixels=sorted(set(supports[0])|set(supports[1]));L=np.array([[s.get(p,0.) for p in pixels] for s in supports])
        np.testing.assert_allclose(R[np.ix_([0,4],[0,4])],noise().rgb_var[0]*L@L.T,atol=1e-18)
        self.assertNotEqual(R[0,4],0);self.assertGreater(np.linalg.eigvalsh(R).min(),0)
        info=J.joint_information(np.zeros(12),np.zeros(12),np.eye(12),[r1,r2],R);self.assertGreater(info['cross_information_norm'],0)

    def test_identical_sensor_events_are_singular_not_jittered(self):
        c=cam();im=textured();r=simple_row(c,np.array([.013,-.007,1.55]),np.ones(3)*.5,im,0);other={**r,'pair_index':1}
        R=J.shared_sensor_covariance([r,other],[im],noise(),POLICY)
        self.assertEqual(np.linalg.matrix_rank(R),4)
        with self.assertRaises(np.linalg.LinAlgError):J.joint_information(np.zeros(12),np.zeros(12),np.eye(12),[r,other],R)

    def test_joint_time_propagation_preserves_cross_point_covariance(self):
        P=np.eye(12);P[0,6]=P[6,0]=.3;P[3,9]=P[9,3]=.2
        s,C,F,Q=J.joint_predict(np.arange(12,dtype=float),P,1/59,np.eye(3)*.01)
        np.testing.assert_allclose(C,F@P@F.T+Q,atol=1e-15);self.assertGreater(np.linalg.norm(C[:6,6:]),0);self.assertGreater(np.linalg.eigvalsh(C).min(),0)

    def test_marginal_assignment_couples_conditionally_independent_points(self):
        means=np.zeros((2,12));means[0,[0,6]]=[1.,0.];means[1,[0,6]]=[0.,1.]
        _,C=J.mixture_moments([.5,.5],means,np.tile(np.eye(12)*.1,(2,1,1)))
        self.assertAlmostEqual(C[0,6],-.25);self.assertGreater(np.linalg.eigvalsh(C).min(),0)
        _,nextC,*_=J.joint_predict(np.zeros(12),C,1/59,np.eye(3)*.01);self.assertNotEqual(nextC[0,6],0)

    def test_proper_toy_mixture_mixed_hessian_finite_difference(self):
        s=np.zeros(12);s[[0,6]]=[.2,-.4];E,g,H,t=J.toy_mixture(s,.1,.5);eps=1e-5
        fd=np.zeros((12,12));dg=np.zeros(12)
        for j in range(12):
            e=np.eye(12)[j]*eps;plus=J.toy_mixture(s+e,.1,.5);minus=J.toy_mixture(s-e,.1,.5)
            dg[j]=(plus[0]-minus[0])/(2*eps);fd[:,j]=(plus[1]-minus[1])/(2*eps)
        np.testing.assert_allclose(g,dg,atol=1e-9);np.testing.assert_allclose(H,fd,atol=1e-8);self.assertNotEqual(H[0,6],0)
        np.testing.assert_allclose(H,H.T,atol=1e-15);self.assertAlmostEqual(t.sum(),1.)

    def test_identical_mixture_components_are_assignment_ambiguous(self):
        s=np.zeros(12);s[[0,6]]=.3;_,_,_,p=J.toy_mixture(s,.2,.1)
        np.testing.assert_array_equal(p,[.5,.5]);_,_,_,q=J.toy_mixture(s,.2,.1,(.9,.1));np.testing.assert_allclose(q,[.9,.1])
        with self.assertRaises(ValueError):J.toy_mixture(s,.2,.1,(1.,1.))

    def test_supported_latent_gaussian_schur_equals_dense_gls(self):
        H=np.zeros((8,12));H[:4,0]=1;H[4:,6]=1;B=np.tile(np.eye(4),(2,1));R=np.eye(8)*.2;C=np.eye(4)*.3
        result=J.nuisance_gls(H,np.arange(8)/10,R,B,C,np.eye(12))
        self.assertLess(result['mean_closure'],1e-12);self.assertLess(result['covariance_closure'],1e-12);self.assertGreater(abs(result['covariance'][0,6]),0)

    def test_same_depth_different_rgb_does_not_supply_pixel_coverage(self):
        c=cam();im=textured();im.depth[:]=1.5;im.rgb[:]=.5
        sc=SimpleNamespace(cams=[c],images=lambda k:[im],x0=np.array([[0.,0.,1.5],[0.,0.,1.5]]),noise=noise(),run=SimpleNamespace(can={'canonical_rgb':np.array([[.5]*3,[.6]*3])}))
        a=J.fixed_pixel_compatibility(sc,0,40,32,0);b=J.fixed_pixel_compatibility(sc,0,40,32,1)
        self.assertNotEqual(a['negative_log_RGB_Z_component'],b['negative_log_RGB_Z_component']);self.assertIsNone(a['complete_assignment_evidence'])
        self.assertEqual(a['location_likelihood'],'UNDEFINED_MODEL');np.testing.assert_array_equal(a['H'][:3],0)
        sc.run.can['canonical_rgb'][1]=.5;b=J.fixed_pixel_compatibility(sc,0,40,32,1);self.assertEqual(a['negative_log_RGB_Z_component'],b['negative_log_RGB_Z_component'])

    def test_invalid_foreground_pixel_is_not_a_candidate_factor(self):
        c=cam();im=textured();im.mask[:]=False
        sc=SimpleNamespace(cams=[c],images=lambda k:[im])
        self.assertEqual(J.fixed_pixel_compatibility(sc,0,40,32,0)['status'],'UNDEFINED_SUPPORT')

    def test_rgb_bias_at_same_material_and_crease_are_separate(self):
        c=cam();im=textured();other=copy.deepcopy(im);other.rgb+=.01;x=np.array([.013,-.007,1.55]);a=np.array([.5]*3)
        r0=simple_row(c,x,a,im,0);r1=simple_row(c,x,a,other,0);np.testing.assert_allclose(r1['r'][:3]-r0['r'][:3],-.01,atol=1e-15)
        # Constant addition cancels analytically; floating subtraction rounds.
        eps=32*np.finfo(float).eps;np.testing.assert_allclose(r1['H'],r0['H'],rtol=eps,atol=eps);u,v,*_=c.project(x);other.depth[int(v),int(u)]+=.5
        r=simple_row(c,x,a,other,0);self.assertEqual(r['sample']['reason'],O.DISCONT)

    def test_policy_has_no_GT_path(self):
        tree=ast.parse((J.REPO/'tools/f1_joint_responsibility_audit.py').read_text())
        for n in ast.walk(tree):
            if isinstance(n,ast.ImportFrom):self.assertNotIn('GroundTruth',[a.name for a in n.names]);self.assertNotIn('evaluate_projective_gt',n.module or '')

    def test_12D_shared_factor_gradient_and_hessian_match_dense_differences(self):
        local=np.r_[np.eye(3),[[.2,.1,.3]]];rows=[{'pair_index':j,'H':local,'r':np.array([.1,.2,-.1,.03])} for j in range(2)]
        H=np.zeros((8,12));H[:4,:3]=local;H[4:,6:9]=local;R=np.kron(np.array([[1.,.3],[.3,1.]]),np.eye(4));P=np.eye(12);s=np.arange(12)*.01;y=np.concatenate([r['r'] for r in rows])
        def evaluate(state):
            current=[{**r,'r':(y-H@state)[4*j:4*j+4]} for j,r in enumerate(rows)]
            return J.joint_information(state,np.zeros(12),P,current,R)
        original=evaluate(s);fd=np.zeros(12);hessian=np.zeros((12,12));eps=1e-5
        for j in range(12):
            d=np.eye(12)[j]*eps;plus=evaluate(s+d);minus=evaluate(s-d);fd[j]=.5*(plus['criterion']-minus['criterion'])/(2*eps);hessian[:,j]=(-plus['b']+minus['b'])/(2*eps)
        np.testing.assert_allclose(fd,-original['b'],atol=1e-10);np.testing.assert_allclose(hessian,original['A'],atol=1e-10)
        self.assertGreater(np.linalg.norm(original['A'][:6,6:]),0)

    def test_source_only_unobserved_moving_point_keeps_predicted_motion(self):
        model=object.__new__(J.PairProblem);model.sm=np.zeros(12);model.sm[3]=.2;model.P=np.eye(12);model.rows_initial=[]
        model.sc=SimpleNamespace(run=SimpleNamespace(fman={'max_iekf_iters':10,'step_tolerance':1e-8}))
        result=model.optimize();self.assertEqual(result['status'],'UNOBSERVED_COAST');self.assertEqual(result['state'][3],.2)

    def test_hash_verification_detects_changed_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'data';p.write_text('original');record=J.file_record(p);inv={'protected':{},'files':{str(p):record}}
            J.verify(inv);p.write_text('changed')
            with self.assertRaisesRegex(RuntimeError,'Input/code changed'):J.verify(inv)


class JointArchivedTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sc=J.Scene(ROOT,J.SCENE);cls.env=J.Environment(cls.sc);cls.problems,cls.original,cls.parity=J.baseline(cls.sc,cls.env)

    def test_G0_both_sources_and_exact_frame1_collision(self):
        import torch
        from estimator.projective_camera import CameraRig
        self.assertEqual(CameraRig(self.sc.run.data.cameras,'cpu',torch.float64).calibration_hash(),self.sc.run.fman['camera_geometry_hash'])
        a=J.source_reprojection(self.sc,8444);b=J.source_reprojection(self.sc,45588)
        self.assertEqual((a['source_camera'],a['source_u'],a['source_v']),(0,282,237));self.assertEqual((b['source_camera'],b['source_u'],b['source_v']),(1,231,232))
        self.assertLess(a['backprojection_error_m'],1e-12);self.assertLess(b['backprojection_error_m'],1e-12)
        occupied=self.env.occupants(0,282,237);self.assertEqual(occupied[0]['point_id'],45588);self.assertIn(8444,[r['point_id'] for r in occupied])

    def test_G1_8444_parity_and_45588_not_fabricated(self):
        self.assertEqual(self.parity[0]['status'],'PARITY_PASS',self.parity)
        self.assertEqual(self.parity[0]['accepted_cameras'],[1]);self.assertEqual(self.parity[1]['status'],'PARITY_NOT_VERIFIABLE')
        self.assertNotIn('P_plus',self.parity[1])
        completeness=J.archive_completeness(self.sc,10);missing=[r for r in completeness['frames'] if r['point_id']==45588 and r['frame']==1][0]
        self.assertFalse(missing['full_6x6_covariance_available']);self.assertIsNotNone(missing['missing_array'])

    def test_G2_transfer_changes_exactly_one_owner_and_keeps_occlusion(self):
        states=np.stack([self.problems[p].s_minus for p in J.PAIR]);Pxx=np.stack([self.problems[p].P_minus[:3,:3] for p in J.PAIR]);x=states[0,:3]
        old=self.env.query(0,x,Pxx,states);new=self.env.query(0,x,Pxx,states,transfer=True)
        self.assertEqual(old['owner'],45588);self.assertEqual(new['owner'],8444)
        other=self.env.query(0,states[1,:3],Pxx,states,transfer=True);self.assertEqual(other['owner'],8444)
        behind=states.copy();cam=self.sc.cams[0];u,v=self.env.slot[1:];z=cam.project(x)[2]+1.
        behind[0,:3]=cam.R.T@(z*np.array([(u-cam.cx)/cam.fx,(v-cam.cy)/cam.fy,1.])-cam.t)
        q=self.env.query(0,behind[0,:3],Pxx,behind,transfer=True);self.assertNotEqual(q['owner'],8444)

    def test_frontmost_order_can_exchange_without_fixed_source_pixels(self):
        states=np.stack([self.problems[p].s_minus for p in J.PAIR]);Pxx=np.stack([self.problems[p].P_minus[:3,:3] for p in J.PAIR])
        before=self.env.query(0,states[0,:3],Pxx,states)['owner'];states[:,:3]=states[::-1,:3].copy()
        after=self.env.query(0,states[0,:3],Pxx,states)['owner'];self.assertEqual(before,45588);self.assertEqual(after,8444)

    def test_G3_same_fixed_observation_set_and_explicit_undefined_evidence(self):
        keys=[(0,282,237),(1,231,232)];enumeration,candidates,counts=J.enumerate_assignments(self.sc,self.env,keys)
        self.assertEqual(len(enumeration),counts['tested_assignments']);self.assertGreater(len(enumeration),1)
        self.assertTrue(all(r['observation_ids']==keys and r['status']=='UNDEFINED_MODEL' for r in enumeration));self.assertTrue(all(r['complete_assignment_evidence'] is None for r in candidates if 'complete_assignment_evidence' in r))

    def test_G5_joint_accepted_steps_descend_without_double_ownership(self):
        U=J.observation_universe(self.sc,self.problems,self.original);model=J.PairProblem(self.sc,self.env,self.problems,U,transfer=True);result=model.optimize()
        for h in result['history']:
            if h['accepted']:self.assertLess(h['criterion_after'],h['criterion_before'])
        claimed=[r['rounded_observation'] for r in result['rows']];self.assertEqual(len(claimed),len(set(claimed)))
        if result['covariance'] is not None:
            np.testing.assert_allclose(result['covariance'],result['covariance'].T,atol=1e-12);self.assertGreater(np.linalg.eigvalsh(result['covariance']).min(),0)

    def test_G6_no_rows_is_coast_not_tracking_success(self):
        U=J.observation_universe(self.sc,self.problems,self.original);model=J.PairProblem(self.sc,self.env,self.problems,U,source_only=True)
        model.rows_initial=[];r=model.optimize();self.assertEqual(r['status'],'UNOBSERVED_COAST');np.testing.assert_array_equal(r['state'],model.sm);np.testing.assert_array_equal(r['covariance'],model.P)

    def test_G7_no_claim_of_full_sequence_with_missing_archive(self):
        archive=J.archive_completeness(self.sc,10);self.assertIn('BLOCKED',archive['sequence_status'])
        self.assertTrue(all(r['tier']=='TIER_UNAVAILABLE' for r in archive['frames'] if r['frame']>1))

    def test_G9_determinism_and_protected_source_hashes(self):
        U=J.observation_universe(self.sc,self.problems,self.original);path=J.REPO/'estimator/point_kalman_cuda.py';h=J.digest(path)
        a=J.PairProblem(self.sc,self.env,self.problems,U,transfer=True).optimize();b=J.PairProblem(self.sc,self.env,self.problems,U,transfer=True).optimize()
        np.testing.assert_array_equal(a['state'],b['state']);self.assertEqual(a['status'],b['status']);self.assertEqual(J.digest(path),h)

    def test_frame1_report_artifacts_and_no_fabricated_sequence(self):
        """Integration smoke test on the actual pair, output confined to /tmp."""
        with tempfile.TemporaryDirectory(prefix='final_joint_integration_') as tmp:
            out=Path(tmp);(out/'figures').mkdir();tests={'successful':True,'tests_run':1,'failures':0,'errors':0,'skips':0}
            J.write_json(out/'tests.json',tests);J.write_json(out/'provenance.json',{'git_commit':'integration-fixture'})
            J.write_json(out/'archive_completeness.json',J.archive_completeness(self.sc,10));J.write_json(out/'noise_and_prior_metadata.json',{'camera_hash_equal':True})
            inv={'protected':{str(p):J.digest(p) for p in J.PROTECTED if p.is_file()},'files':{}}
            code=J.analyze(ROOT,ROOT/'fifth_pass_source_responsibility_20261030',out,tests,inv,10)
            self.assertEqual(code,0);self.assertTrue((out/'FINAL_JOINT_RESPONSIBILITY_CAUSAL_REPORT.md').exists())
            self.assertTrue((out/'figures/storage_8444_45588_camera0_camera1_ownership.png').exists());self.assertTrue((out/'joint_covariance_12x12.npz').exists())
            self.assertFalse((out/'joint_sequence_0_to_10.csv').exists());self.assertEqual(J.read_json(out/'results.json')['complete_fixed_pixel_model'],'UNDEFINED_MODEL')
            self.assertEqual(J.read_json(out/'joint_sequence_status.json')['earliest_unverified_pair_posterior_frame'],1)


if __name__=='__main__':unittest.main()
