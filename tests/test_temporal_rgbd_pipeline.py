import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from tests.test_temporal_warp import rig,frame
from estimator.temporal_run import run_temporal_free
from estimator.temporal_noise import load_warp_noise
from scene.config import load_config
import train

CPU_CUDA_PIPELINE_COMPARISONS=[]


class TinyDataset:
    def __init__(self):
        rr=rig();self.cameras=[SimpleNamespace(name=rr.names[c],width=8,height=8,intrinsics=rr.K[c].numpy(),near=.01,world_to_camera=rr.T_cw[c]) for c in range(2)]
        self.times=np.array([0.,.04,.09,.13]);self.frame_ids=[0,1,2,3]
        self.root=Path('/synthetic-unit-test-not-a-real-scene')
    def observation(self,k,c):
        f=frame();mask=np.zeros((8,8),bool);mask[2:5,2:5]=True
        return SimpleNamespace(camera=self.cameras[c],rgb=f.rgb[c].numpy(),depth=np.full((8,8),2.),mask=mask,raw_valid=np.ones((8,8),bool),valid=mask)


class PipelineTests(unittest.TestCase):
    def test_temporal_sensor_path_does_not_use_cross_camera_mismatch(self):
        from unittest.mock import patch
        from estimator.temporal_noise import temporal_sensor_only
        cfg=load_config();policy=json.loads(Path('configs/projective_rgbd.json').read_text())
        fixture={'rgb_variance':[.01,.02,.03],'depth_variance_m2':.002,'depth_squared_variance':.001,'method':'explicit unit-test sensor, not production calibration'}
        with patch('scene.noise_calibration.temporal_sensor_noise',return_value=fixture),patch('scene.noise_calibration.cross_camera_mismatch',side_effect=AssertionError('cross-camera mismatch is not warp noise')):
            c,p,report=temporal_sensor_only(TinyDataset(),cfg,policy)
        np.testing.assert_array_equal(np.diag(c.color_covariance),fixture['rgb_variance'])
        self.assertEqual(c.provenance['depth_variance']['class'],'estimated-from-input-under-assumptions')
        self.assertEqual(report['cross_camera_mismatch'],'NOT_COMPUTED_OR_USED_FOR_TEMPORAL_MODEL')
    def test_one_invalid_point_does_not_freeze_an_independent_update(self):
        from estimator.temporal_simplified_iekf import update_points_temporal_simplified
        from estimator.temporal_zbuffer import first_hit_pixel_owner
        rr=rig();f=frame();mean=torch.tensor([[-.5,0,2,0,0,0],[.5,0,2,0,0,0]],dtype=torch.float64)
        P=torch.eye(6,dtype=torch.float64).repeat(2,1,1)
        class TestRows(SimpleNamespace):
            def __len__(self):return len(self.point)
        def build(s):
            ids=torch.tensor([0,1]) if s[0,0]==mean[0,0] else torch.tensor([1])
            H=torch.zeros((len(ids),4,6),dtype=torch.float64);H[:,0,0]=1.
            rows=TestRows(point=ids,camera=torch.zeros(len(ids),dtype=torch.long),pixel=ids+20,reference_frame=torch.zeros(len(ids),dtype=torch.long),
              H_t=H,y=torch.cat(((mean[ids,:1]+.1),torch.zeros((len(ids),3),dtype=torch.float64)),1),
              h=torch.cat((s[ids,:1],torch.zeros((len(ids),3),dtype=torch.float64)),1),R=torch.eye(4,dtype=torch.float64).repeat(len(ids),1,1))
            own=first_hit_pixel_owner(s[:,:3],rr,f)
            return rows,own,own.reason,{}
        state,cov,*_,solver=update_points_temporal_simplified(mean,P,build,f,max_iters=2)
        self.assertEqual(float(state[0,0]),-.5);self.assertGreater(float(state[1,0]),.54)
        torch.testing.assert_close(cov[0],P[0],atol=0,rtol=0)
        self.assertLess(float(cov[1,0,0]),1.)
    def test_focal_controls_preserve_original_identity(self):
        cfg=load_config();policy=json.loads(Path('configs/projective_rgbd.json').read_text())
        cov,doc=load_warp_noise('/not-read','ideal')
        options={'mode':'simplified','lag':3,'warp_covariance':cov.tolist(),'warp_provenance':doc,'workspace_bytes':128*1024**2,'diagnostic_original_point_ids':[4,9]}
        sub={'factor':1,'method':'spatial-stratified','seed':0,'boundary_preserve':False,'report':False}
        with tempfile.TemporaryDirectory() as tmp:
            run_temporal_free(TinyDataset(),cfg,policy,'forward',tmp,'cpu',sub,options,[],[4,9],max_iters=2)
            can=np.load(Path(tmp)/'canonical.npz');np.testing.assert_array_equal(can['point_id'],[4,9])
            np.testing.assert_array_equal(can['subsampled_to_full_index'],[4,9])
            sm=json.loads((Path(tmp)/'subsample_manifest.json').read_text())
            self.assertEqual(sm['N_selected'],2);self.assertEqual(sm['N_full'],18)
            self.assertTrue(sm['diagnostic_focal_control'])
            self.assertTrue((Path(tmp)/'temporal/traced_points/9.json').exists())

    def test_calibration_zero_difference_is_finite_and_not_invented(self):
        from tools.calibrate_temporal_warp_noise import robust_covariance,disjoint_temporal_pairs
        result=robust_covariance(np.zeros((40,4)))
        np.testing.assert_array_equal(result['covariance'],np.zeros((4,4)))
        self.assertEqual(result['effective_samples'],40.)
        train,heldout,excluded=disjoint_temporal_pairs([8,0,1,2,5])
        self.assertEqual(excluded,[2])
        self.assertFalse({f for k in train for f in (k,k+1)}&{f for k in heldout for f in (k,k+1)})

    def test_large_covariance_diagnostics_preserve_independent_matrices(self):
        from estimator.temporal_run import batched_eigvalsh
        for device in ['cpu']+(['cuda'] if torch.cuda.is_available() else []):
            matrix=torch.eye(3,device=device,dtype=torch.float64).repeat(70001,1,1)
            result=batched_eigvalsh(matrix)
            torch.testing.assert_close(result,torch.ones((70001,3),device=device,dtype=torch.float64),atol=0,rtol=0)
    def test_ideal_warp_keeps_sensor_noise_and_full_lag_covariance(self):
        cov,doc=load_warp_noise('/not-a-calibration-folder','ideal')
        self.assertEqual(doc['assumption_class'],'IDEAL_WARP_ASSUMPTION_UNCALIBRATED')
        np.testing.assert_array_equal(cov,np.zeros((4,4)))
        cfg=load_config();policy=json.loads(Path('configs/projective_rgbd.json').read_text())
        sub={'factor':1,'method':'spatial-stratified','seed':0,'boundary_preserve':False,'report':False}
        for mode in ('full','simplified'):
            with self.subTest(mode=mode),tempfile.TemporaryDirectory() as tmp:
                options={'mode':mode,'lag':3,'warp_covariance':cov.tolist(),'warp_provenance':doc,'workspace_bytes':128*1024**2}
                run_temporal_free(TinyDataset(),cfg,policy,'forward',tmp,'cpu',sub,options,[],[0],max_iters=3)
                P=np.load(Path(tmp)/'position_variance_diag.npy')
                self.assertTrue((P>0).all())
                records=json.loads((Path(tmp)/'temporal/traced_points/0.json').read_text())['frames']
                self.assertTrue(any(len(r['rows']) for r in records))

    def test_full_no_new_observations_is_exact_prediction(self):
        from estimator.point_kalman_cuda import transition,process_noise
        class NoLaterMasks(TinyDataset):
            def observation(self,k,c):
                obs=super().observation(k,c)
                if k:obs.mask[:]=False;obs.valid[:]=False
                return obs
        cfg=load_config();policy=json.loads(Path('configs/projective_rgbd.json').read_text())
        options={'mode':'full','lag':3,'warp_covariance':np.eye(4).tolist(),'warp_provenance':{'class':'test-only'},'workspace_bytes':128*1024**2}
        sub={'factor':1,'method':'spatial-stratified','seed':0,'boundary_preserve':False,'report':False}
        with tempfile.TemporaryDirectory() as tmp:
            run_temporal_free(NoLaterMasks(),cfg,policy,'forward',tmp,'cpu',sub,options,[],[0],max_iters=2)
            trace=json.loads((Path(tmp)/'temporal/traced_points/0.json').read_text())['frames']
            for record in trace:
                np.testing.assert_array_equal(record['s_minus'],record['s_plus'])
                np.testing.assert_array_equal(record['P_minus'],record['P_plus'])

    def test_forensic_imports_do_not_disable_cuda(self):
        import subprocess,sys
        # Also exercise the programmatic forensic CLI gate used by full discovery.
        code="import os; from unittest import TestLoader; before=os.environ.get('CUDA_VISIBLE_DEVICES'); import tools.f1_source_responsibility_audit,tools.f1_joint_responsibility_audit; from tests import test_f1_definitive_drift_forensics as m; suite=TestLoader().loadTestsFromModule(m); test=next(t for group in suite for t in group if t._testMethodName=='test_all_stops_before_analysis_when_cpu_tests_error'); test.debug(); assert os.environ.get('CUDA_VISIBLE_DEVICES')==before"
        subprocess.run([sys.executable,'-B','-c',code],check=True)

    def test_cli_defaults_and_invalid_combinations(self):
        _,a=train.parse(['--scene','/tmp/not-read'])
        self.assertEqual(a.observation_model,'legacy')
        for args in (['--temporal-lag-frames','1'],['--observation-model','temporal-rgbd','--free-solver','temporal-ieks'],
                     ['--observation-model','temporal-rgbd','--rendered-mask-measurement','diagnostic']):
            with self.assertRaises(SystemExit):train.parse(['--scene','/tmp/not-read',*args])

    def test_missing_warp_and_explicit_provenance(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError,'MISSING_TEMPORAL_WARP_NOISE'):load_warp_noise(tmp)
            p=Path(tmp)/'warp.json';p.write_text(json.dumps({'covariance_rgbd':np.diag([.01,.01,.01,.001]).tolist(),
                'assumption_class':'synthetic-fixture-only','method':'declared exact test generative model','units':['encoded_rgb_0_1']*3+['m']}))
            cov,doc=load_warp_noise(tmp,'config',p)
            self.assertEqual(doc['assumption_class'],'synthetic-fixture-only')
            with self.assertRaisesRegex(ValueError,'measured warp policy'):load_warp_noise(tmp,'measured',p)

    def test_both_modes_both_directions_four_frame_slide_and_html(self):
        config=load_config();policy=json.loads(Path('configs/projective_rgbd.json').read_text())
        sub={'factor':1,'method':'spatial-stratified','seed':0,'boundary_preserve':False,'report':False}
        for mode in ('simplified','full'):
            for direction in ('forward','backward'):
                baseline=None
                for device in ['cpu']+(['cuda'] if torch.cuda.is_available() else []):
                    with self.subTest(mode=mode,direction=direction,device=device),tempfile.TemporaryDirectory() as tmp:
                        result=self.check_pipeline(mode,direction,device,tmp,config,policy,sub)
                        if device=='cpu':baseline=result
                        else:
                            errors={key:float(np.max(np.abs(result[key]-baseline[key]))) for key in result}
                            CPU_CUDA_PIPELINE_COMPARISONS.append({'mode':mode,'direction':direction,**errors})
                            for key in result:np.testing.assert_allclose(result[key],baseline[key],atol=2e-9,rtol=2e-7,err_msg=key)

    def check_pipeline(self,mode,direction,device,tmp,config,policy,sub):
        options={'mode':mode,'lag':3,'warp_covariance':np.diag([.01,.01,.01,.001]).tolist(),
                 'warp_provenance':{'class':'synthetic-fixture-only'},'workspace_bytes':128*1024**2}
        status=run_temporal_free(TinyDataset(),config,policy,direction,Path(tmp),device,sub,options,[0,1,3],[0,4],max_iters=2)
        self.assertEqual(status['status'],'COMPLETE')
        X=np.load(Path(tmp)/'posterior_xyz.npy');P=np.load(Path(tmp)/'position_variance_diag.npy')
        self.assertEqual(X.shape,(4,18,3));self.assertTrue(np.isfinite(X).all());self.assertTrue((P>0).all())
        self.assertTrue((Path(tmp)/'temporal/index.html').is_file())
        self.assertTrue((Path(tmp)/'temporal/debug_frames/frame_0001_cam0.png').is_file())
        man=json.loads((Path(tmp)/'run_manifest.json').read_text())
        self.assertEqual(man['processing_order'],list(range(4)) if direction=='forward' else list(range(3,-1,-1)))
        if mode=='full':
            log=json.loads((Path(tmp)/'temporal/marginalization.json').read_text());self.assertTrue(len(log)>0)
        trace=json.loads((Path(tmp)/'temporal/traced_points/0.json').read_text())['frames'][-1]
        return {'xyz_max_abs':X,'position_diagonal_max_abs':P,'traced_covariance_max_abs':np.asarray(trace['P_plus'])}

    def test_train_cli_both_modes_and_existing_html_exports(self):
        from tests.projective_fixtures import write_scene
        for mode in ('full','simplified'):
            with self.subTest(mode=mode),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);scene=write_scene(TinyDataset(),root/'scene')
                (scene/'metadata/timing.json').write_text(json.dumps({'timestamps_seconds':TinyDataset().times.tolist()}))
                warp=root/'warp.json';warp.write_text(json.dumps({'covariance_rgbd':np.diag([.01,.01,.01,.001]).tolist(),
                    'assumption_class':'synthetic-fixture-only','method':'declared fixture only, never production default','units':['encoded_rgb_0_1']*3+['m']}))
                out=root/'run'
                rc=train.main(['--scene',str(scene),'--output',str(out),'--device','cpu','--direction','both','--stop-after','free',
                    '--sensor-noise','config','--observation-model','temporal-rgbd','--temporal-mode',mode,'--temporal-warp-noise','config',
                    '--temporal-warp-noise-config',str(warp),'--debug','light','--debug-frames','0,1,3','--max-iekf-iters','2','--no-evaluate','--no-qa'])
                self.assertEqual(rc,0)
                self.assertTrue((out/'debug/free_forward_3d.html').is_file());self.assertTrue((out/'debug/free_backward_rgbd.html').is_file())
                self.assertIn('Temporal forward',(out/'debug/index.html').read_text())
