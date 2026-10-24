import json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from numpy.testing import assert_allclose,assert_array_equal
from grouped_fixtures import GroupScene
from pointwise_rgbd.config import load_config
from pointwise_rgbd.group_process import load_group_noise
from pointwise_rgbd.phase1_grouped_controller import run_phase1
from rgbd_common.backends.torch_cuda import Backend
from rgbd_common.array_store import ArrayStore


class GroupControllerTests(unittest.TestCase):
    def test_cli_timing_overrides_reach_selected_dataset_and_provenance(self):
        import contextlib,io
        from data_fixtures import make_dataset
        from tools.run_grouped_rgbd_joint import main
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);scene=root/'scene';scene.mkdir();make_dataset(scene,frames=4,cameras=2)
            (scene/'metadata/job_config_resolved.yaml').write_text('fps: null\nduration_sec: null\nvideo_fps: null\n')
            timestamps=root/'timestamps.json';timestamps.write_text(json.dumps([0,.02,.07,.15]))
            for name,flags,expected in [('fps',['--fps','20'],[0,.05,.1]),('dt',['--dt-seconds','.025'],[0,.025,.05]),('timestamps',['--timestamps-file',str(timestamps)],[0,.02,.07])]:
                output=root/name
                with self.subTest(name=name),contextlib.redirect_stdout(io.StringIO()),patch('pointwise_rgbd.free_phase1.run',return_value=(None,None)) as filtering,patch('pointwise_rgbd.group_visualization.export_viewer'):
                    code=main(['--scene',str(scene),'--output',str(output),'--frame-limit','3','--stop-after','phase1','--device','cpu','--anchor','first','--no-viewer',*flags])
                self.assertEqual(code,3)
                data=filtering.call_args.args[0];assert_allclose(data.times,expected)
                manifest=json.loads((output/'manifest.json').read_text())
                self.assertEqual(manifest['dataset']['timing']['source_kind'],'explicit-user-override')
                self.assertEqual(manifest['dataset']['timing']['interval_count'],3)
            with contextlib.redirect_stderr(io.StringIO()) as stderr,patch('pointwise_rgbd.free_phase1.run') as filtering:
                code=main(['--scene',str(scene),'--output',str(root/'missing'),'--frame-limit','3','--stop-after','phase1','--device','cpu','--anchor','first','--no-viewer'])
            self.assertEqual(code,2);filtering.assert_not_called()
            self.assertIn('no FPS was assumed',stderr.getvalue());self.assertNotIn('Traceback',stderr.getvalue())
            self.assertIn('job_config_resolved.yaml',(root/'missing/failure.log').read_text())
            with patch('pointwise_rgbd.free_phase1.run') as filtering,self.assertRaisesRegex(ValueError,'inconsistent user timing'):
                main(['--scene',str(scene),'--output',str(root/'conflicting'),'--frame-limit','3','--stop-after','phase1','--device','cpu','--anchor','first','--no-viewer','--fps','20','--dt-seconds','.1'])
            filtering.assert_not_called()

    def test_static_both_directions_gauges_unique_factors_exact_rigidity_resume(self):
        c=load_config();noise=load_group_noise();b=Backend('cpu');scene=GroupScene()
        with tempfile.TemporaryDirectory() as temporary:
            for anchor,k0 in [('first',0),('last',7)]:
                root=Path(temporary)/anchor
                with patch('pointwise_rgbd.controller.run_exact_pointwise_discovery',side_effect=AssertionError('recovery forbidden')):
                    partition,a=run_phase1(scene,c,noise,b,root,anchor)
                expected=np.repeat(np.eye(4)[None,None],8,axis=0);expected=np.repeat(expected,partition.count,axis=1)
                assert_allclose(a['group_poses'],expected,atol=1e-14)
                assert_array_equal(a['group_poses'][k0],expected[k0]);assert_allclose(a['covariance'][k0,:,:6,:],0)
                factors=ArrayStore(root/'actual_measurement_factors.sqlite')
                for k in range(8):
                    for group in range(partition.count):
                        ids=np.flatnonzero(partition.membership==group)
                        assert_allclose(np.linalg.norm(a['xyz'][k,ids]-a['xyz'][k,ids[0]],axis=1),np.linalg.norm(partition.xyz[ids]-partition.xyz[ids[0]],axis=1),atol=2e-14)
                    if k!=k0:
                        refs=factors.read(str(k));pairs=np.c_[refs['camera'],refs['pixel']]
                        self.assertEqual(len(np.unique(pairs,axis=0)),len(pairs))
                snapshots={f.name:f.read_bytes() for f in root.glob('*.npy')}
                run_phase1(scene,c,noise,b,root,anchor,resume=True)
                self.assertEqual(snapshots,{f.name:f.read_bytes() for f in root.glob('*.npy')})
                metadata=json.loads((root/'run.json').read_text());self.assertEqual(metadata['chronological_original_frames'],scene.frame_ids)
    def test_interrupted_checkpoint_resume_equals_uninterrupted_history(self):
        from pointwise_rgbd.group_iekf import correct
        scene=GroupScene(frames=6,motion='translation');c=load_config();noise=load_group_noise();b=Backend('cpu');calls=[0]
        def interrupted(*args,**kwargs):
            calls[0]+=1
            if calls[0]==3:raise RuntimeError('deliberate interrupted frame')
            return correct(*args,**kwargs)
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            with patch('pointwise_rgbd.phase1_grouped_controller.correct',side_effect=interrupted):
                with self.assertRaisesRegex(RuntimeError,'deliberate interrupted'):
                    run_phase1(scene,c,noise,b,root/'interrupted')
            _,resumed=run_phase1(scene,c,noise,b,root/'interrupted',resume=True)
            _,fresh=run_phase1(scene,c,noise,b,root/'fresh')
            for name in ('group_poses','twists','covariance','xyz','point_observed'):
                assert_array_equal(resumed[name],fresh[name])

    def test_partial_and_twenty_frame_complete_occlusion_preserve_rigidity_and_uncertainty(self):
        scene=GroupScene(frames=40,motion='translation',occlusion=True);c=load_config();b=Backend('cpu')
        for k in range(15,35):
            frame=scene.observation(k,0);frame.mask[:]=False
        with tempfile.TemporaryDirectory() as root:
            partition,a=run_phase1(scene,c,load_group_noise(),b,root,target_occupancy=1000)
            self.assertTrue((a['group_measurements'][15:35]==0).all())
            self.assertTrue((a['pose_rank'][15:35]==0).all())
            self.assertGreater(float(np.trace(a['covariance'][34,0,:6,:6])),float(np.trace(a['covariance'][14,0,:6,:6])))
            reference=np.linalg.norm(partition.xyz-partition.xyz[0],axis=1)
            for k in range(40):assert_allclose(np.linalg.norm(a['xyz'][k]-a['xyz'][k,0],axis=1),reference,atol=2e-13)

    def test_full_source_is_not_blocked_by_discarded_physical_splitting_gate(self):
        from test_data_loading import real_scene
        from tools.run_grouped_rgbd_joint import main
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);scene=root/'scene';scene.mkdir();real_scene(scene)
            with patch('pointwise_rgbd.free_phase1.run',return_value=(None,None)) as filtering:
                code=main(['--scene',str(scene),'--output',str(root/'blocked'),'--stop-after','phase1','--device','cpu','--anchor','first','--no-viewer'])
            self.assertEqual(code,3)
            result=json.loads((root/'blocked/result.json').read_text())
            self.assertEqual(result['status'],'INCOMPLETE_OR_UNRESOLVED');filtering.assert_called_once();self.assertTrue(result['full_source']);self.assertFalse(result['complete'])

    def test_metadata_input_pointer_cannot_write_inside_resolved_source(self):
        from test_data_loading import real_scene
        from tools.run_grouped_rgbd_joint import main
        with tempfile.TemporaryDirectory() as temporary:
            scene=Path(temporary)/'scene';scene.mkdir();real_scene(scene);output=scene/'must_not_exist'
            with self.assertRaises(SystemExit) as error:
                main(['--scene',str(scene/'metadata'),'--output',str(output),'--frame-limit','2','--stop-after','phase1','--device','cpu','--anchor','first','--no-viewer'])
            self.assertEqual(error.exception.code,2);self.assertFalse(output.exists())

    def test_translation_group_transport_tracks_measured_xyz(self):
        scene=GroupScene(motion='translation');c=load_config();b=Backend('cpu')
        with tempfile.TemporaryDirectory() as root:
            partition,a=run_phase1(scene,c,load_group_noise(),b,root,target_occupancy=64)
            expected=partition.xyz+scene.truth[-1][:3,3]
            self.assertLess(np.linalg.norm(a['xyz'][-1]-expected,axis=1).mean(),.015)
            self.assertTrue((a['group_measurements'][1:]>0).all())
            self.assertTrue((a['pose_rank'][1:]==6).all())


if __name__=='__main__':unittest.main()
