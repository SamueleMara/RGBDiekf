"""Section 9 source-responsibility tests: CPU float64, no production writes."""
import ast
import copy
from dataclasses import replace
import hashlib
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
from tools import root_cause_oracle as O
from tools import f1_source_responsibility as S
from tools.f1_definitive_evidence import DEFAULT_ROOT,REPO
from tools.f1_definitive_math import exact_information,fixed_cost
from tests.test_f1_temporal_reference_diagnostics import cam,textured,noise
from tests.projective_fixtures import POLICY

ROOT=Path(os.environ.get('F1_SOURCE_FORENSICS_ROOT',str(DEFAULT_ROOT)))


def fixture(v=None,frozen=None,images=None):
    cameras=[cam(),cam()];images=images or [textured(),textured(np.array([.002,-.001,.001]))]
    x=np.array([.0131,-.0077,1.55]);u,w,*_=cameras[0].project(x);a=O.sample(images[0],cameras[0],u+.02,w,noise(),POLICY)['rgb']
    sm,Pm,*_=O.predict(np.r_[x,0.,0.,0.],np.diag([1e-4]*3+[1.]*3),1/59,np.eye(3)*.01)
    refs={c:{'valid':True,'b':np.array([.001,.001,.001,.0001]),'var0':np.array([1e-5]*3+[1e-8]),'status':'VALID_REFERENCE'} for c in range(2)}
    f0=[{'frame0_class':'SOURCE_ID_KNOWN_AT_FRAME0' if c==0 else 'FRAME0_GEOMETRICALLY_PLAUSIBLE','material_identity':'SOURCE_ID_KNOWN_AT_FRAME0' if c==0 else 'MATERIAL_ID_UNDECIDABLE'} for c in range(2)]
    p=S.ResponsibilityProblem(cameras,images,a,sm,Pm,frozen or [O.VALID,O.VALID],noise(),POLICY,10,1e-8,refs=refs,
        variant=v or S.Variant('A_ALL_RAW'),source=0,frame0=f0)
    return p,f0,refs


class SourceResponsibilityUnitTests(unittest.TestCase):
    def test_01_source_backprojection_and_reference_dependency(self):
        camera=cam();im=textured();u,v=40,32;D=im.depth[v,u];x=camera.R.T@(D*np.array([(u-camera.cx)/camera.fx,(v-camera.cy)/camera.fy,1.])-camera.t)
        sc=SimpleNamespace(src=np.array([[0,0,v*camera.W+u]]),x0=np.array([x]),cams=[camera],
             run=SimpleNamespace(can={'subsampled_to_full_index':np.array([9]),'canonical_rgb':np.array([im.rgb[v,u]])},data=SimpleNamespace(observation=lambda k,c:im)))
        r=S.source_reprojection(sc,0);self.assertLess(r['backprojection_error_m'],1e-14);self.assertEqual(r['full_endpoint_index'],9)
        row=O.row(im,camera,x,im.rgb[v,u],noise(),POLICY,np.eye(3)*1e-4,np.zeros(3));np.testing.assert_allclose(row['r'],0,atol=1e-13)
        self.assertIn('not independent',r['source_correlation'])

    def test_02_point_identity_does_not_freeze_pixels(self):
        p,_,_=fixture(S.Variant('B_SOURCE_RAW','source'));x=p.s_minus[:3];moved=x+np.array([.005,0,0]);u0=p.rows(x)[0]['u'];u1=p.rows(moved)[0]['u']
        self.assertNotEqual(u0,u1);self.assertEqual(p.rows(x)[0]['camera'],p.rows(moved)[0]['camera'])
        with self.assertRaises(ValueError):S.eligibility_mask([{}],0,S.Variant('INVALID','fixed_frame0_pixel'))

    def test_04_source_eligibility_changes_no_noise_prior_or_gate(self):
        p,f0,refs=fixture();v=S.Variant('B_SOURCE_RAW','source');q,_,_=fixture(v)
        a=p.rows(p.s_minus[:3]);b=q.rows(q.s_minus[:3]);self.assertEqual([r['camera'] for r in q.used(b)],[0])
        for key in ('H','r','var','S','nu'):np.testing.assert_array_equal(a[0][key],b[0][key])
        np.testing.assert_array_equal(p.P_minus,q.P_minus);self.assertEqual(p.policy,q.policy);self.assertEqual(p.max_iters,q.max_iters)

    def test_05_probe_never_overrides_physical_rejection(self):
        p,_,_=fixture(frozen=[O.SELF_OCC,O.VALID]);self.assertEqual(S.source_probe(p,0,p.s_minus[:3])['reason'],O.SELF_OCC)
        for field,reason in [('mask',O.MASK),('raw_valid',O.INVALID_DEPTH)]:
            p,_,_=fixture(frozen=[O.DUP,O.VALID]);getattr(p.imgs[0],field)[:]=False;self.assertEqual(S.source_probe(p,0,p.s_minus[:3])['reason'],reason)
        p,_,_=fixture(frozen=[O.DUP,O.VALID]);self.assertEqual(S.source_probe(p,0,np.array([100.,0.,1.]))['reason'],O.OUTSIDE)
        p.imgs[0].depth[:]=1.;self.assertEqual(S.source_probe(p,0,p.s_minus[:3])['reason'],O.FRONT)

    def test_06_owner_provenance_and_single_slot_transfer(self):
        p,_,_=fixture(S.Variant('SRC_PRIORITY_COUNTERFACTUAL',source_priority=True),[O.DUP,O.VALID]);x=p.s_minus[:3]
        self.assertEqual(p.rows(x)[0]['reason'],O.VALID);self.assertEqual(p.frozen[0],O.DUP)
        source=np.array([[0,0,1],[0,1,2]]);sc=SimpleNamespace(src=source,cams=p.cams,x0=np.array([x,x]),policy=p.policy,
              run=SimpleNamespace(can={'canonical_rgb':np.array([p.a,p.a]),'subsampled_to_full_index':np.array([10,11])}))
        probe={'Z':1.55,'sigma_z2_prior':1e-4,'pixel_occupants_total':2,'pixel_occupants_front_to_back':[{'point':1,'z':1.549,'sigma_z2_prior':1e-4},{'point':0,'z':1.55,'sigma_z2_prior':1e-4}]}
        r=S.owner_from_probe(sc,probe,0);self.assertEqual(r['owner_point_id'],1);self.assertEqual(r['owner_source']['source_camera'],1)
        assignment={1:0};assignment.pop(r['owner_point_id']);assignment[0]=0;self.assertEqual(len(assignment),1)

    def test_07_missing_reference_retains_raw_or_omits_explicitly(self):
        p,_,_=fixture(S.Variant('D_ALL_REFERENCED',reference_channels='RGBD'));p.refs[1]={'valid':False,'b':None,'var0':None,'status':'NO_VALID_REFERENCE'}
        row=p.rows(p.s_minus[:3])[1];ordinary=O.row(p.imgs[1],p.cams[1],p.s_minus[:3],p.a,p.noise,p.policy,p.P_minus[:3,:3],np.zeros(3))
        np.testing.assert_array_equal(row['r'],ordinary['r']);self.assertFalse(row['reference_applied'])
        p.variant=replace(p.variant,missing_reference='omit');self.assertNotIn(1,[r['camera'] for r in p.used(p.rows(p.s_minus[:3]))])

    def test_08_geometry_is_not_material_verification_and_ambiguity_is_factorial(self):
        p,f0,_=fixture();row=p.rows(p.s_minus[:3])[1];self.assertEqual(S.classify_frame0(row,O.VALID,False),'FRAME0_GEOMETRICALLY_PLAUSIBLE')
        self.assertTrue(S.eligibility_mask(f0,0,S.Variant('C','frame0',ambiguous='admit'))[1]);self.assertFalse(S.eligibility_mask(f0,0,S.Variant('C','frame0',ambiguous='exclude'))[1])
        self.assertEqual(f0[1]['material_identity'],'MATERIAL_ID_UNDECIDABLE')

    def test_10_no_motion_with_no_rows_is_not_tracking_success(self):
        self.assertEqual(S.tracking_assessment([{'observed':False}],'moving'),'MOVING_TRACKING_FAILED_NO_OBSERVATIONS')

    def test_11_source_probe_is_not_a_second_assimilation(self):
        p,_,_=fixture(frozen=[O.DUP,O.VALID]);before=p.run(POLICY['update_acceptance']);r=S.source_probe(p,0,p.s_minus[:3]);after=p.run(POLICY['update_acceptance'])
        self.assertEqual(r['reason'],O.VALID);self.assertFalse(r['assimilated']);np.testing.assert_array_equal(before['P_plus'],after['P_plus']);np.testing.assert_array_equal(before['s_plus'],after['s_plus'])

    def test_12_invalid_fixed_support_is_undefined(self):
        p,_,_=fixture();s=p.s_minus.copy();s[0]=100.;result=fixed_cost(p,s,[0]);self.assertEqual(result['status'],'UNDEFINED_SUPPORT');self.assertIsNone(result['cost'])

    def test_13_finite_difference_and_common_hessian_force_closure(self):
        p,_,_=fixture();x=p.s_minus[:3];check=O.finite_difference_H(p,x,0,1e-8);self.assertFalse(check['crosses_pixel_cell']);self.assertLess(check['rel_error'],2e-5)
        rows=p.used(p.rows(x));info=exact_information(p.s_minus,p.s_minus,p.P_minus,rows);A,b,d,*_=p.solve(p.s_minus,rows)
        np.testing.assert_allclose(info['A'],A,rtol=1e-12);np.testing.assert_allclose(info['b'],b,rtol=1e-12);np.testing.assert_allclose(info['contributions'].sum(0),d,atol=1e-12)

    def test_14_policy_has_no_GT_import_or_static_label_input(self):
        tree=ast.parse((REPO/'tools/f1_source_responsibility.py').read_text())
        for node in ast.walk(tree):
            if isinstance(node,ast.ImportFrom):self.assertNotIn('evaluation',node.module or '');self.assertNotIn('GroundTruth',[a.name for a in node.names])
        p,_,_=fixture();before=p.run(POLICY['update_acceptance'])
        with patch('tools.forensic_f1_drift.GroundTruth',side_effect=AssertionError('GT entered policy')):after=p.run(POLICY['update_acceptance'])
        np.testing.assert_array_equal(before['s_plus'],after['s_plus'])

    def test_15_deterministic_read_only_point_experiment(self):
        p,_,_=fixture();prior=p.P_minus.copy();images=[hashlib.sha256(im.rgb.tobytes()+im.depth.tobytes()+im.mask.tobytes()).hexdigest() for im in p.imgs]
        a=p.run(POLICY['update_acceptance']);b=p.run(POLICY['update_acceptance']);np.testing.assert_array_equal(a['s_plus'],b['s_plus']);np.testing.assert_array_equal(a['P_plus'],b['P_plus']);np.testing.assert_array_equal(prior,p.P_minus)
        self.assertEqual(images,[hashlib.sha256(im.rgb.tobytes()+im.depth.tobytes()+im.mask.tobytes()).hexdigest() for im in p.imgs])

    def test_17_identical_depth_of_different_patches_is_undecidable(self):
        images=[textured(),textured()];images[0].rgb[:]=.2;images[1].rgb[:]=.8
        p,f0,_=fixture(images=images);row=p.rows(p.s_minus[:3])[1]
        self.assertEqual(f0[1]['material_identity'],'MATERIAL_ID_UNDECIDABLE')
        # Foreground and depth can be identical despite distinct material appearance.
        np.testing.assert_array_equal(images[0].depth,images[1].depth);np.testing.assert_array_equal(images[0].mask,images[1].mask)
        self.assertNotEqual(images[0].rgb[0,0,0],images[1].rgb[0,0,0]);self.assertNotIn('VERIFIED_MATERIAL_ID',S.classify_frame0(row,O.VALID,False))

    def test_18_moving_source_projection_can_become_mask_occluded(self):
        p,_,_=fixture(S.Variant('B_SOURCE_RAW','source'));x=p.s_minus[:3];u0=p.cams[0].project(x)[0];moved=x+np.array([.03,0.,0.]);u,v,*_=p.cams[0].project(moved)
        p.imgs[0].mask[int(v):int(v)+2,int(u):int(u)+2]=False
        self.assertNotEqual(u0,u);self.assertEqual(p.rows(moved)[0]['reason'],O.MASK);self.assertFalse(p.used(p.rows(moved)))

    def test_source_priority_never_bypasses_true_occlusion(self):
        p,_,_=fixture(S.Variant('SRC_PRIORITY_COUNTERFACTUAL',source_priority=True),[O.SELF_OCC,O.VALID]);self.assertEqual(p.rows(p.s_minus[:3])[0]['reason'],O.SELF_OCC)
        p,_,_=fixture(S.Variant('SRC_PRIORITY_COUNTERFACTUAL',source_priority=True),[O.DUP,O.VALID]);p.imgs[0].mask[:]=False;self.assertEqual(p.rows(p.s_minus[:3])[0]['reason'],O.DUP)

    def test_six_arm_family_and_variance_missing_ambiguity_factors(self):
        variants=S.variants();self.assertTrue({'A_ALL_RAW','B_SOURCE_RAW','C_FRAME0_PLAUSIBLE_RAW','D_ALL_REFERENCED','E_FRAME0_PLAUSIBLE_REFERENCED','F_SOURCE_REFERENCED'}<={v.name for v in variants})
        self.assertEqual(len({S.variant_id(v) for v in variants}),len(variants))
        self.assertEqual({v.ambiguous for v in variants if v.eligibility=='frame0'},{'admit','exclude'})

    def test_chunked_population_feature_pipeline_on_two_measured_points(self):
        from tests.projective_fixtures import FieldDataset,camera
        from tools.f1_source_responsibility_population import population_features,score_population
        im=textured();c=camera('c0',K=(400.,400.,47.5,35.5),W=96,H=72);oc=O.Cam(c);pixels=[(40,32),(45,35)]
        xyz=np.array([oc.R.T@(im.depth[v,u]*np.array([(u-oc.cx)/oc.fx,(v-oc.cy)/oc.fy,1.])-oc.t) for u,v in pixels]);P0=np.tile(np.eye(3)*1e-4,(2,1,1))
        ds=FieldDataset([c],[[(im.rgb,im.depth,im.mask)]]*11);ds.times=np.arange(11)/59.
        arrays={'posterior_xyz':np.tile(xyz,(11,1,1)),'velocity':np.zeros((11,2,3)),'point_status':np.ones((11,2),np.uint8),'camera_reason':np.zeros((11,1,2),np.uint8)}
        run=SimpleNamespace(data=ds,can={'canonical_rgb':np.array([im.rgb[v,u] for u,v in pixels])},mm=lambda name:arrays[name],fman={'prior':{'velocity_covariance_m2_s2':np.eye(3),'q_c_m2_s3':np.eye(3)*.01}})
        with tempfile.TemporaryDirectory(prefix='fifth_pass_source_fixture_') as tmp:
            sc=SimpleNamespace(name='fixture',x0=xyz,src=np.array([[0,0,v*96+u] for u,v in pixels]),cams=[oc],P0=P0,run=run,policy=POLICY,noise=noise(),t=ds.times,order=list(range(11)),fdir=Path(tmp))
            features=population_features(sc,tmp,chunk=1);np.savez(Path(tmp)/'population_events.npz',x0=xyz,gt_static=np.ones(2,bool))
            self.assertEqual(len(features),2);self.assertTrue(features['source_admitted'].all());self.assertTrue(np.isfinite(features['full_hessian_next_dx_m']).all())
            scored=score_population(sc,features,tmp);self.assertEqual(sum(r['N'] for r in scored),2);self.assertTrue(all(r['fraction_drift_gt_1cm_frame10']==0 for r in scored))


class SourceResponsibilityArchivedTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from tools.f1_second_pass import Scene
        cls.sc={s:Scene(ROOT,s) for s in S.CASES};cls.own={s:S.Frame0Ownership(sc) for s,sc in cls.sc.items()};cls.cache={}

    def point(self,scene,pid):
        key=(scene,pid)
        if key not in self.cache:self.cache[key]=S.frame0_audit(self.sc[scene],pid,self.own[scene])
        return self.sc[scene],*self.cache[key]

    def test_03_A_all_raw_primary_and_archived_stable_covariance_parity(self):
        with np.load(ROOT/'fourth_pass_sequential_replay_20261030/sequence_arrays.npz',allow_pickle=False) as fourth:
            for scene,pid,N in [('storage_45135',8444,10),('USB_100109',67827,10),('storage_45135',60019,5)]:
                sc,f0,refs=self.point(scene,pid);frames=S.replay(sc,pid,S.Variant('A_ALL_RAW'),f0,refs,N);rows=S.baseline_parity(sc,pid,frames,fourth)
                self.assertTrue(all(r['status']=='PARITY_PASS' for r in rows),(scene,pid,rows));self.assertEqual(len(rows),N)

    def test_07_USB_camera4_missing_reference_and_actual_frame1_cameras(self):
        sc,f0,refs=self.point('USB_100109',67827);self.assertFalse(refs[4]['valid']);self.assertIsNone(refs[4]['b'])
        frames=S.replay(sc,67827,S.Variant('A_ALL_RAW'),f0,refs,1);self.assertEqual(frames[0]['rows_accepted'],[5,6])
        temporal=S.temporal_audit(sc,67827,refs,1)
        missing=[r for r in temporal if r['camera']==4]
        self.assertTrue(all(r['delta_r'] is None for r in missing));self.assertTrue(all(r['reference_status']=='NO_VALID_REFERENCE' for r in missing))

    def test_09_RGB_depth_reference_causal_channels_without_retuning(self):
        for scene,pid,prefer in [('storage_45135',8444,'RGB'),('USB_100109',67827,'D')]:
            sc,f0,refs=self.point(scene,pid);norms={}
            for channels in ('RGB','D'):
                frames=S.replay(sc,pid,S.Variant('REFERENCE',reference_channels=channels),f0,refs,1);norms[channels]=float(np.linalg.norm(frames[0]['dx_m']))
            self.assertLess(norms[prefer],norms['D' if prefer=='RGB' else 'RGB'],(scene,pid,norms))

    def test_10_static_neighbor_and_unobserved_moving_points(self):
        sc,f0,refs=self.point('USB_100109',51974);frames=S.replay(sc,51974,S.Variant('A_ALL_RAW'),f0,refs,5)
        self.assertLessEqual(max(np.linalg.norm(f['s_plus'][:3]-sc.x0[51974]) for f in frames),S.POSITION_TOL)
        for pid in (87290,85696):
            sc,f0,refs=self.point('USB_100109',pid);frames=S.replay(sc,pid,S.Variant('B_SOURCE_RAW','source'),f0,refs,5)
            self.assertNotIn('SUCCESS',S.tracking_assessment(frames,'moving'))

    def test_16_8416_parity_failure_is_indeterminate_with_separate_units(self):
        sc,f0,refs=self.point('storage_45135',8416);rows=S.baseline_parity(sc,8416,S.replay(sc,8416,S.Variant('A_ALL_RAW'),f0,refs,1))
        self.assertEqual(rows[0]['status'],'PARITY_FAIL');self.assertIn('position_linf_m',rows[0]);self.assertIn('velocity_linf_m_s',rows[0])

    def test_source_owner_is_actual_archived_PID_with_complete_provenance(self):
        sc=self.sc['storage_45135'];probe=sc.trace(1)['iterations'][0]['points']['8444']['cameras'][0];owner=S.owner_from_probe(sc,probe,8444)
        self.assertEqual(probe['visibility_frozen'],'DUPLICATE_PIXEL_SUPPRESSED');self.assertNotEqual(owner['owner_point_id'],8444)
        self.assertEqual(owner['owner_source']['source_camera'],int(sc.src[owner['owner_point_id'],1]))

    def test_extraction_factorial_serialization_and_report_integration(self):
        from tools.f1_source_responsibility_audit import TABLES,audit_archived_update,record_frames,checkpoint
        from tools.report_f1_source_responsibility import generate
        from tools.f1_definitive_evidence import write_json
        sc,f0,refs=self.point('storage_45135',8444);rows={name:[] for name in TABLES};arrays={}
        with tempfile.TemporaryDirectory(prefix='fifth_pass_source_integration_') as tmp:
            out=Path(tmp);(out/'figures').mkdir();audit_archived_update(sc,8444,1,rows,arrays,out)
            self.assertTrue(any(r['is_source'] and r['owner_point_id']!=8444 for r in rows['source_owner_conflicts']))
            self.assertTrue(any(k.endswith('__production_A') for k in arrays))
            v=S.Variant('A_ALL_RAW');base=S.replay(sc,8444,v,f0,refs,1);event=record_frames(sc,8444,v,base,base,rows,True,'CONTINUOUS')
            self.assertEqual(rows['sequence_state'][0]['frame'],0);self.assertEqual(event['first_source_loss'],1)
            checkpoint(out,rows,arrays,[{'scene':sc.name,'point_id':8444,'source':S.source_record(sc,8444),'parity':'PARITY_PASS','decision':'INDETERMINATE','arm_outcomes':[event]}],['fixture only'],{'successful':True})
            write_json(out/'tests.json',{'successful':True});write_json(out/'evidence_inventory.json',{'git_commit':'fixture'})
            text=generate(out).read_text();self.assertIn('MATERIAL_ID_UNDECIDABLE',text);self.assertIn('8444',text);self.assertTrue((out/'figures/storage_45135_8444_1_owner.png').exists())


if __name__=='__main__':unittest.main()
