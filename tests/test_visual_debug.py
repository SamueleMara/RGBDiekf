import tempfile,unittest
from pathlib import Path
import numpy as np
from parallel_fixtures import Scene,configuration
from rgbd_common.backends.torch_cuda import Backend
from rgbd_common.fusion import fuse_frame_zero
from rgbd_common.lie import State
from rgbd_common.outputs import display_indices,live_view


class VisualDebugTests(unittest.TestCase):
    def test_small_group_visible_and_offline_live_score_export(self):
        model,_=fuse_frame_zero(Scene(2,1),configuration(),Backend('cpu'))
        model.group[-1]=1
        ids=display_indices(model,10)
        self.assertEqual(len(ids),10);self.assertEqual(len(set(ids)),10)
        self.assertEqual(set(model.group[ids]),{0,1})
        states=[State(np.tile(np.eye(4),(2,1,1)),np.zeros((2,6))) for _ in range(2)]
        records=[{'elapsed_seconds':1.,'proposal_energy':100.,'accepted':True},
                 {'elapsed_seconds':2.,'proposal_energy':101.,'accepted':False}]
        with tempfile.TemporaryDirectory() as root:
            reports=Path(root)/'reports';reports.mkdir()
            live_view(reports,model,states,'Accepted <map>',1,records)
            index=(Path(root)/'visual_debug/index.html').read_text()
            motion=(Path(root)/'visual_debug/motion.html').read_text()
            self.assertIn('Accepted &lt;map&gt;',index);self.assertIn('101.0',index)
            self.assertIn('Estimated rigid group',motion)
            self.assertTrue((Path(root)/'visual_debug/plotly.min.js').is_file())
            self.assertFalse((Path(root)/'visual_debug/motion.pending.html').exists())
