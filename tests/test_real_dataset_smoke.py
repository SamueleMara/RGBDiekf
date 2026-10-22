"""Read the COMPLETE explicitly selected scene; this is input integrity, not SLAM success."""
import json,os,unittest
from pathlib import Path
import numpy as np
from rgbd_common.dataloader import load_dataset
from rgbd_common.diagnostics import validate_full_dataset


@unittest.skipUnless(os.environ.get('RGBD_REAL_SCENE'),'Set RGBD_REAL_SCENE for the mandatory complete-scene input test')
class RealDatasetTests(unittest.TestCase):
    def test_every_frame_camera_depth_mask_calibration_and_timing(self):
        dataset=load_dataset(Path(os.environ['RGBD_REAL_SCENE']))
        report=validate_full_dataset(dataset)
        self.assertEqual(report['frames'],len(dataset.times));self.assertEqual(report['cameras'],len(dataset.cameras))
        self.assertTrue(np.all(np.diff(dataset.times)>0));self.assertIsNotNone(dataset.masks)
        self.assertLess(report['source_reprojection_max_pixel_error'],1e-7)
        self.assertFalse(report['inference_ground_truth_used'])
        output=os.environ.get('RGBD_TEST_REPORTS')
        if output:
            root=Path(output);root.mkdir(parents=True,exist_ok=True)
            (root/'real_dataset_integrity.json').write_text(json.dumps(report,indent=2)+'\n')
