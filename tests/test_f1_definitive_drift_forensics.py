"""CPU fifth-pass oracles. Real-data tests never run without archived inputs."""
import ast,copy,hashlib,os,tempfile,unittest,warnings
from types import SimpleNamespace
from unittest.mock import patch
from pathlib import Path
import numpy as np
from scipy.linalg import block_diag
from tools import root_cause_oracle as O
from tools import f1_definitive_math as M
from tools.f1_definitive_evidence import DEFAULT_ROOT,MANDATORY,REPO,make_inventory,write_json
from tools.f1_definitive_point import replay,baseline_parity,FrozenSupport,run_control,dense_objective_oracle
from tests.test_f1_temporal_reference_diagnostics import cam,noise,textured,Img
from tests.projective_fixtures import POLICY

ROOT=Path(os.environ.get('F1_DEFINITIVE_ROOT',str(DEFAULT_ROOT)))


class DefinitiveMathTests(unittest.TestCase):
    def fixture(self):
        c=cam();im=textured();s=np.r_[.0131,-.0077,1.55,0,0,0];P=np.diag([1e-4]*3+[1.]*3);sm,Pm,*_=O.predict(s,P,1/59,np.eye(3)*.01)
        u,v,*_=c.project(sm[:3]);a=O.sample(im,c,u+.05,v+.02,noise(),POLICY)['rgb']
        return O.Problem([c],[im],a,sm,Pm,[O.VALID],noise(),POLICY,12,1e-8)

    def test_short_historical_archive_keeps_all_ten_production_comparisons(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);archive=root/'fourth_pass_sequential_replay_20261030';archive.mkdir()
            key='storage_7__A0_ORIGINAL_F1__P__';state=np.zeros(6);P=np.eye(6)
            np.savez(archive/'sequence_arrays.npz',**{key+'s_plus':np.zeros((5,6)),key+'P_plus':np.tile(P,(5,1,1))})
            final={'s_plus':state,'P_plus':P,'acceptance':'ACCEPTED_FINAL_ITERATE','rows':[{'camera':1}],'converged':True}
            prediction={'points':{'7':{'s_minus':state,'P_minus':P}},'F':P,'Q':np.zeros((6,6))}
            sc=SimpleNamespace(name='storage_45135',run=SimpleNamespace(root=root/'storage_45135/traced'),order=list(range(11)),trace=lambda k:{'final':{'7':final},'prediction':prediction})
            frames=[{'frame':k,'s':state,'P':P,'sm':state,'Pm':P,'F':P,'Q':np.zeros((6,6)),'rows':[1],'res':{'acceptance_code':1,'converged':True}} for k in range(1,11)]
            rows,failed=baseline_parity(sc,7,frames)
            self.assertIsNone(failed);self.assertEqual(len(rows),10)
            self.assertTrue(all(r['fourth_pass_comparison']['status']=='PARITY_PASS' for r in rows[:5]))
            self.assertTrue(all(r['fourth_pass_comparison']['status']=='NO_ARCHIVED_FRAME' for r in rows[5:]))
            # Missing historical data must never conceal a production mismatch.
            frames[-1]={**frames[-1],'s':state+np.array([.01,0,0,0,0,0])}
            rows,failed=baseline_parity(sc,7,frames);self.assertEqual(failed,10);self.assertEqual(rows[-1]['status'],'PARITY_FAIL')

    def test_all_stops_before_analysis_when_cpu_tests_error(self):
        from tools import f1_definitive_drift_forensics as runner
        with patch.object(runner,'make_inventory',return_value={'scenes':{},'git_commit':'fixture'}),patch.object(runner,'run_tests',return_value={'successful':False,'errors':1}),patch.object(runner,'analyze') as analyze,patch('tools.report_f1_definitive_drift.generate_test_gate_failure',return_value=Path('/tmp/fifth_pass_failure_fixture/TEST_GATE_FAILURE.md')) as report:
            self.assertEqual(runner.main(['all','--out','/tmp/fifth_pass_failure_fixture']),1);analyze.assert_not_called()
            report.assert_called_once()

    def test_failed_gate_report_needs_no_analysis_and_cannot_claim_success(self):
        from tools.report_f1_definitive_drift import generate_test_gate_failure
        with tempfile.TemporaryDirectory() as tmp:
            path=generate_test_gate_failure(tmp,{'tests_run':1,'failures':1,'errors':0,'skips':0,'records':[{'test':'fixture.parity','status':'FAIL','detail':'Covariance parity failed'}]})
            text=path.read_text();self.assertIn('No accepted causal',text);self.assertIn('Covariance parity failed',text)
            with self.assertRaises(FileExistsError):generate_test_gate_failure(tmp,{})

    def test_covariance_block_diagnostics_never_relax_original_parity_gate(self):
        P=np.eye(6);P[0,1]=P[1,0]=1e-14;other=P.copy();other[0,1]+=3e-11;other[1,0]+=3e-11
        comparison=M.covariance_agreement(other,P)
        self.assertFalse(comparison['covariance_equal']);self.assertTrue(comparison['covariance_block_norm_equal']);self.assertFalse(comparison['covariance_elementwise_equal'])
        self.assertAlmostEqual(comparison['covariance_block_checks']['xx']['frobenius_error'],np.sqrt(2)*3e-11)
        other=P.copy();other[0,3]=other[3,0]=1e-8
        self.assertFalse(M.covariance_agreement(other,P)['covariance_equal'],'A zero cross-block still has a strict absolute bound')
        other=P.copy();other[3:,3:]*=1.001
        self.assertFalse(M.covariance_agreement(other,P)['covariance_equal'])

    def test_solver_warnings_and_invalid_final_support_are_reported(self):
        p=self.fixture()
        def failed_solver(*args,**kw):
            warnings.warn('fixture: invalid Powell trial',RuntimeWarning)
            return SimpleNamespace(x=np.array([100.,100.,1.]),success=True,message='fixture')
        with patch('tools.f1_definitive_point.minimize',side_effect=failed_solver):result=dense_objective_oracle(p,[0])
        self.assertEqual(result['status'],'NUMERICAL_ORACLE_FAILED');self.assertFalse(result['success']);self.assertIsNone(result['cost_after'])
        self.assertGreater(result['invalid_support_trials'],0);self.assertIn('fixture: invalid Powell trial',result['warnings'])

    def test_G2a_six_state_rhs_hessian_and_contributions_close(self):
        p=self.fixture();r=p.used(p.rows(p.s_minus[:3]));a=M.exact_information(p.s_minus,p.s_minus,p.P_minus,r);A,b,d,*_=p.solve(p.s_minus,r)
        np.testing.assert_allclose(a['A'],A,rtol=1e-12);np.testing.assert_allclose(a['b'],b,rtol=1e-12);np.testing.assert_allclose(a['delta'],d,rtol=1e-11,atol=1e-12)
        np.testing.assert_allclose(a['contributions'].sum(0),d,atol=1e-12);self.assertLess(a['attribution_closure'],1e-12)
        self.assertTrue((a['dimensionless_eigenvalues']>0).all())

    def test_G2a_high_precision_is_same_operator_not_a_new_pixel_branch(self):
        p=self.fixture();r=p.used(p.rows(p.s_minus[:3]));a=M.exact_information(p.s_minus,p.s_minus,p.P_minus,r)
        reference=M.high_precision_step(a['A'],a['b'])
        np.testing.assert_allclose(reference,a['delta'],rtol=1e-10,atol=1e-12)
        self.assertLess(np.linalg.norm(a['A']@reference-a['b'])/max(np.linalg.norm(a['b']),1.),1e-12)

    def test_G2a_camera_subset_attribution_is_order_invariant(self):
        p=self.fixture();row=p.used(p.rows(p.s_minus[:3]))[0];second={**row,'camera':1,'r':row['r']*.4}
        a=M.subset_attribution(p.s_minus,p.s_minus,p.P_minus,[row,second]);b=M.subset_attribution(p.s_minus,p.s_minus,p.P_minus,[second,row])
        self.assertLess(a['closure'],1e-12);np.testing.assert_allclose(a['contributions'],b['contributions'][::-1],atol=1e-12)

    def test_G0b_excluded_samples_cannot_enter_force_attribution(self):
        p=self.fixture();r=p.used(p.rows(p.s_minus[:3]))[0];r={**r,'reason':O.DUP}
        with self.assertRaisesRegex(ValueError,'Excluded'):M.exact_information(p.s_minus,p.s_minus,p.P_minus,[r])

    def test_G2b_exact_velocity_transfer_and_signed_process(self):
        p=self.fixture();rows=p.used(p.rows(p.s_minus[:3]));a=M.exact_information(p.s_minus,p.s_minus,p.P_minus,rows)
        gain,e=M.velocity_transfer(p.P_minus,a['delta'][:3],a['delta'][3:]);np.testing.assert_allclose(e,0,atol=1e-11)
        for dt in (1/59,-1/59):
            _,P,F,Q=O.predict(np.zeros(6),np.eye(6),dt,np.eye(3)*.01)
            np.testing.assert_allclose(P,F@F.T+Q,atol=1e-15);self.assertEqual(np.sign(Q[0,3]),np.sign(dt))

    def test_G2c_invalid_support_is_undefined(self):
        p=self.fixture();s=p.s_minus.copy();s[0]=100.
        r=M.fixed_cost(p,s,[0]);self.assertEqual(r['status'],'UNDEFINED_SUPPORT');self.assertIsNone(r['cost'])

    def test_G2d_frozen_image_can_drive_nonzero_update_without_temporal_change(self):
        p=self.fixture();r=p.run(POLICY['update_acceptance']);q=self.fixture();q.imgs=p.imgs
        s=q.run(POLICY['update_acceptance']);np.testing.assert_array_equal(r['s_plus'],s['s_plus']);self.assertTrue(r['observed']);self.assertGreater(np.linalg.norm(r['s_plus'][:3]-p.s_minus[:3]),0)
        for field in ('rgb','depth','mask'):np.testing.assert_array_equal(getattr(p.imgs[0],field),getattr(q.imgs[0],field))

    def test_G2e_actual_single_time_velocity_position_nullspace(self):
        p=self.fixture();H=p.used(p.rows(p.s_minus[:3]))[0]['H'];dt=1/59
        design=np.c_[H,dt*H];N=np.vstack([-dt*np.eye(3),np.eye(3)])
        np.testing.assert_allclose(design@N,0,atol=1e-14)
        out=M.observability([np.c_[H,np.zeros((4,3))]],[np.block([[np.eye(3),dt*np.eye(3)],[np.zeros((3,3)),np.eye(3)]])])
        self.assertEqual(out['status'],'NOT_IDENTIFIABLE');self.assertGreater(out['nullspace'].shape[1],0)

    def test_G2f_schur_joint_and_correlated_gls_match(self):
        g=np.random.default_rng(724);H=g.normal(size=(12,6));B=np.tile(np.eye(4),(3,1));R=np.diag(g.uniform(.1,1,12));C=np.eye(4)*.4;P=np.eye(6);y=g.normal(size=12)
        r=M.nuisance_gls(H,y,R,B,C,P)
        for key in ('mean_closure','covariance_closure','joint_closure'):self.assertLess(r[key],1e-12)
        naive=np.linalg.inv(np.eye(6)+H.T@np.linalg.solve(np.diag(np.diag(r['shared_covariance'])),H));self.assertGreater(np.linalg.norm(naive-r['covariance']),1e-3)
        raw=block_diag(C,*[np.eye(4)*.2 for _ in range(3)]);S=M.differenced_covariance(raw,3,4)
        np.testing.assert_array_equal(S[:4,4:8],C)

    def test_G1a_backprojection_sample_center_and_source_correlation(self):
        c=cam();im=textured();u,v=40,32;z=float(im.depth[v,u]);x=c.R.T@(z*np.array([(u-c.cx)/c.fx,(v-c.cy)/c.fy,1])-c.t)
        a=im.rgb[v,u].copy();r=O.row(im,c,x,a,noise(),POLICY,np.eye(3)*1e-4,np.zeros(3))
        np.testing.assert_allclose(r['r'],0,atol=1e-13);s=O.sample(im,c,float(u),float(v),noise(),POLICY)
        np.testing.assert_array_equal(s['rgb'],a);self.assertEqual(s['depth'],z)
        # The source pixel defines BOTH coordinates and appearance; not two independent observations.
        pairs=[]
        for e in (-1e-3,0.,1e-3):
            D=z+e;source=c.R.T@(D*np.array([(u-c.cx)/c.fx,(v-c.cy)/c.fy,1])-c.t);pairs.append([(c.R@source+c.t)[2],D])
        covariance=np.cov(np.asarray(pairs).T);self.assertGreater(covariance[0,1],0)
        self.assertLess(np.var(np.asarray(pairs)[:,0]-np.asarray(pairs)[:,1]),1e-25)
        self.assertGreater(covariance[0,0]+covariance[1,1],0)

    def test_G1b_differentiable_cells_both_sides_of_integer(self):
        c=cam();im=textured();a=np.array([.4,.5,.6])
        for u in (40.-1e-3,40.+1e-3):
            z=1.55;x=c.R.T@(z*np.array([(u-c.cx)/c.fx,(32.3-c.cy)/c.fy,1])-c.t)
            p=O.Problem([c],[im],a,np.r_[x,0,0,0],np.diag([1e-4]*3+[1.]*3),[O.VALID],noise(),POLICY,8,1e-8)
            fd=O.finite_difference_H(p,x,0,eps=1e-8);self.assertFalse(fd['crosses_pixel_cell']);self.assertLess(fd['rel_error'],2e-5)

    def test_G1c_surface_allowance_and_full_joint_nis(self):
        p=self.fixture();r=p.rows(p.s_minus[:3])[0];sample=r['sample'];z=np.mean(sample['neighbour_depth']);expected=POLICY['same_surface_n_sigma']*np.sqrt(2*noise().sigma2(z))+z*np.tan(np.radians(POLICY['same_surface_max_incidence_deg']))*np.sqrt(2)/min(p.cams[0].fx,p.cams[0].fy)
        self.assertAlmostEqual(sample['same_surface_allowance_m'],expected,places=14)
        S=r['H']@p.P_minus[:3,:3]@r['H'].T+np.diag(r['var']);np.testing.assert_allclose(S,r['S'],atol=1e-15)
        self.assertAlmostEqual(float(r['nu']@np.linalg.solve(S,r['nu'])),r['nis'],places=10)

    def test_G1d_raw_equality_is_not_reference_validity(self):
        from tools.f1_temporal_reference_diagnostics import references
        from tests.test_f1_temporal_reference_diagnostics import stub_scene
        im=textured();im.mask[:]=False;sc=stub_scene(im);ref=references(sc,7,[O.VALID])[0]
        self.assertTrue(np.array_equal(im.rgb,im.rgb.copy()));self.assertFalse(ref['valid']);self.assertIsNone(ref['b'])

    def test_G3a_position_velocity_units_and_branch_parity_are_independent(self):
        a=np.zeros(6);b=a.copy();b[3]=.646;r=M.parity(a,b,np.eye(6),np.eye(6),[1],[1],'A','A')
        self.assertEqual(r['position_error_linf_m'],0);self.assertEqual(r['velocity_error_linf_m_s'],.646);self.assertEqual(r['status'],'PARITY_FAIL')
        p=M.branch_coordinates(np.nextafter(40.,-np.inf),30.);self.assertEqual(p['u']['floor'],39);self.assertNotEqual(p['u']['hex'],float(40).hex())

    def test_G3c_missing_reference_is_never_zero_imputed(self):
        from tools.f1_temporal_reference_diagnostics import ref_row
        p=self.fixture();x=p.s_minus[:3];ordinary=O.row(p.imgs[0],p.cams[0],x,p.a,p.noise,p.policy,p.P_minus[:3,:3],np.zeros(3))
        ref={'valid':False,'b':None,'var0':None,'status':'NO_VALID_REFERENCE'}
        modified=ref_row(p.imgs[0],p.cams[0],x,p.a,p.noise,p.policy,p.P_minus[:3,:3],np.zeros(3),ref,'RGBD')
        np.testing.assert_array_equal(ordinary['r'],modified['r']);self.assertFalse(modified['reference_applied'])

    def test_G2g_controls_repeat_and_do_not_mutate_images_or_original_problem(self):
        p=self.fixture();before=[getattr(p.imgs[0],k).copy() for k in ('rgb','depth','mask')];prior=p.P_minus.copy()
        a=run_control(p,POLICY['update_acceptance']);b=run_control(p,POLICY['update_acceptance'])
        np.testing.assert_array_equal(a['s_plus'],b['s_plus']);np.testing.assert_array_equal(p.P_minus,prior)
        for k,v in zip(('rgb','depth','mask'),before):np.testing.assert_array_equal(getattr(p.imgs[0],k),v)
        regular=p.run(POLICY['update_acceptance']);np.testing.assert_allclose(a['s_plus'],regular['s_plus'],atol=1e-14)

    def test_G3e_production_does_not_import_definitive_diagnostic_paths(self):
        names=('f1_definitive','report_f1_definitive')
        for folder in ('estimator','scene'):
            for p in (REPO/folder).glob('*.py'):
                tree=ast.parse(p.read_text())
                for node in ast.walk(tree):
                    if isinstance(node,ast.ImportFrom):self.assertFalse(any(s in (node.module or '') for s in names),(p,node.lineno))
                    elif isinstance(node,ast.Import):self.assertFalse(any(any(s in x.name for s in names) for x in node.names),(p,node.lineno))

    def test_fresh_output_refuses_overwrite(self):
        with tempfile.TemporaryDirectory(prefix='fifth_pass_') as tmp:
            with self.assertRaises(FileExistsError):make_inventory(ROOT,Path(tmp))

    def test_depth_plane_never_claims_trusted_face_identity(self):
        p=self.fixture();r=M.plane_patch(p.imgs[0].depth,p.imgs[0].mask,p.imgs[0].raw_valid,40,32,5,p.cams[0],p.noise,.01)
        self.assertEqual(r['face_status'],'FACE_IDENTITY_UNPROVEN')


class DefinitiveArchivedDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from tools.f1_second_pass import Scene
        cls.sc={name:Scene(ROOT,name) for name in MANDATORY}

    def test_G0a_exact_sequential_required_points_and_covariance(self):
        for name,pid in [('storage_45135',8444),('USB_100109',67827),('USB_100109',51975),('USB_100109',51974),('storage_45135',60019)]:
            rows,failed=baseline_parity(self.sc[name],pid,replay(self.sc[name],pid,10))
            self.assertEqual(len(rows),10);self.assertIsNone(failed,(name,pid,rows));self.assertTrue(all(r['status']=='PARITY_PASS' for r in rows))

    def test_G0b_actual_admitted_camera_errata(self):
        from tools.f1_second_pass import problem
        for name,pid,excluded in [('USB_100109',67827,{4}),('storage_45135',262894,{3,4}),('blade_103706',14104,{0})]:
            sc=self.sc[name];p=problem(sc,1,pid);r=p.run(sc.policy['update_acceptance']);used={x['camera'] for x in p.used(r['rows_accepted'])}
            self.assertFalse(used&excluded);prod={x['camera'] for x in sc.trace(1)['final'][str(pid)].get('rows',[])};self.assertEqual(used,prod)
        self.assertIn(6,{x['camera'] for x in self.sc['USB_100109'].trace(1)['final']['67827']['rows']})

    def test_G0c_catalogue_baseline_mapping_timestamp_and_calibration(self):
        from tools.root_cause_trace import Run
        from tools.f1_definitive_evidence import paths_for
        for name,sc in self.sc.items():
            baseline=Run(paths_for(ROOT,name)['baseline'],'forward')
            for key in ('point_id','subsampled_to_full_index','original_source_ids','canonical_xyz','canonical_rgb'):np.testing.assert_array_equal(sc.run.can[key],baseline.can[key])
            self.assertEqual(sc.order,baseline.fman['processing_order']);self.assertEqual(sc.run.fman['timestamps_seconds'],baseline.fman['timestamps_seconds']);self.assertEqual(sc.run.fman['camera_geometry_hash'],baseline.fman['camera_geometry_hash'])
            np.testing.assert_array_equal(sc.run.data.times,sc.t)
            for cam,archived in zip(sc.run.data.cameras,sc.run.man['cameras']['cameras']):
                self.assertEqual(cam.name,archived['name']);np.testing.assert_array_equal(cam.intrinsics,archived['intrinsics_fx_fy_cx_cy']);np.testing.assert_allclose(cam.world_to_camera,archived['T_cw'],rtol=0,atol=1e-14)

    def test_G3a_known_8416_nonparity_never_becomes_position_units_bug(self):
        sc=self.sc['storage_45135'];rows,failed=baseline_parity(sc,8416,replay(sc,8416,3))
        self.assertIsNotNone(failed,'If this branch now matches, investigate/version the branch before changing the known-erratum expectation.')
        r=rows[failed-1];self.assertGreater(r['position_error_l2_m'],1e-4);self.assertGreater(r['velocity_error_l2_m_s'],1e-3)

    def test_G3b_accepted_nonconverged_and_increasing_cost_are_not_silenced(self):
        from tools.f1_second_pass import problem
        nonconv=[];increasing=[]
        for pid,k in ((49199,1),(67801,2),(102354,1),(102354,3)):
            sc=self.sc['storage_45135'];f=sc.trace(k)['final'][str(pid)];p=problem(sc,k,pid);cams=[r['camera'] for r in f.get('rows',[])]
            if f['acceptance'].startswith('ACCEPTED') and not f['converged']:nonconv.append((pid,k))
            before=M.fixed_cost(p,p.s_minus,cams);after=M.fixed_cost(p,np.asarray(f['s_plus']),cams)
            if before['status']==after['status']=='DEFINED' and after['cost']>before['cost']:increasing.append((pid,k))
        self.assertTrue(nonconv);self.assertTrue(increasing,'The prescribed counterexample must be rechecked against identical valid support, not silently dropped.')

    def test_G3d_GT_cannot_modify_moving_replay_or_masks(self):
        import tools.forensic_f1_drift as F
        previous=F.GroundTruth
        def forbidden(*args,**kw):raise AssertionError('GT entered a replay')
        F.GroundTruth=forbidden
        try:
            for pid in (87290,85696):
                sc=self.sc['USB_100109'];frames=replay(sc,pid,5);self.assertEqual(len(frames),5)
                if pid==85696:
                    for f in frames[2:]:
                        if all(c in (O.DUP,O.SELF_OCC) for c in f['prob'].frozen):self.assertFalse(f['res']['observed'])
        finally:F.GroundTruth=previous


if __name__=='__main__':unittest.main()
