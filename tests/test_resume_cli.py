import contextlib,io,json,tempfile,unittest
from pathlib import Path
import yaml
import numpy as np
from PIL import Image
from test_data_loading import real_scene
from rgbd_common.entry import run
from iekf.filter import IEKF


class ResumeCLITests(unittest.TestCase):
    def test_resume_detects_mask_content_changes_with_unchanged_paths(self):
        with tempfile.TemporaryDirectory() as root:
            source=Path(root)/'scene';source.mkdir();real_scene(source);output=Path(root)/'result'
            args=['--input',str(source),'--output',str(output),'--device','cpu','--validate-only']
            with contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(run(IEKF,args),0);(output/'checkpoint.json').write_text('{}')
                Image.fromarray(np.array([[0,255],[255,255]],np.uint8)).save(source/'masks/cam_000/frame_000000.png')
                self.assertEqual(run(IEKF,args+['--resume']),1)
            self.assertIn('CHECKPOINT_OBSERVATIONS_CHANGED',json.loads((output/'validation_report.json').read_text())['reason'])
    def test_both_entries_reach_estimation_without_visibility_configuration(self):
        from unittest.mock import patch
        from ukf.filter import UKF
        with tempfile.TemporaryDirectory() as root:
            source=Path(root)/'scene';source.mkdir();real_scene(source)
            for path in (source/'masks').rglob('*.png'):
                Image.fromarray(np.array([[0,255],[255,255]],np.uint8)).save(path)
            for estimator in (IEKF,UKF):
                output=Path(root)/estimator.name;terminal=io.StringIO()
                with patch('rgbd_common.entry.run_outer',side_effect=RuntimeError('ESTIMATION_REACHED')) as outer, contextlib.redirect_stdout(terminal),contextlib.redirect_stderr(io.StringIO()):
                    code=run(estimator,['--input',str(source),'--output',str(output),'--device','cpu'])
                self.assertEqual(code,1);outer.assert_called_once()
                report=json.loads((output/'validation_report.json').read_text())
                self.assertEqual(report['reason'],'ESTIMATION_REACHED')
                self.assertNotIn('timestamps_seconds',terminal.getvalue());self.assertNotIn('files_searched',terminal.getvalue())
                resolved=yaml.safe_load((output/'run_config_resolved.yaml').read_text())
                self.assertEqual(resolved['foreground']['hard_mask_model_version'],2)
                self.assertIsNone(resolved['foreground']['mask_visibility_premise']['object_only_no_external_occluders'])
                self.assertTrue((output/'reports/input_assumptions_and_provenance.json').is_file())

    def test_serialized_observation_tuples_do_not_look_like_changed_dataset(self):
        with tempfile.TemporaryDirectory() as root:
            source=Path(root)/'scene';source.mkdir();real_scene(source)
            for rgb in (source/'rgb').rglob('*.png'):
                mask=source/'masks'/rgb.relative_to(source/'rgb');mask.parent.mkdir(parents=True,exist_ok=True)
                Image.fromarray(np.ones((2,2),np.uint8)*255).save(mask)
            output=Path(root)/'result'
            args=['--input',str(source),'--output',str(output),'--device','cpu','--validate-only']
            with contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(run(IEKF,args),0)
                (output/'checkpoint.json').write_text('{}')
                self.assertEqual(run(IEKF,args+['--resume']),0)
                path=output/'run_config_resolved.yaml';previous=yaml.safe_load(path.read_text())
                previous['scene']['observations'][0][0][0]='changed_rgb.png'
                path.write_text(yaml.safe_dump(previous))
                self.assertEqual(run(IEKF,args+['--resume']),1)
            failure=json.loads((output/'validation_report.json').read_text())
            self.assertIn('CHECKPOINT_DATASET_CHANGED: observations',failure['reason'])
