import tempfile,unittest
import numpy as np
from grouped_fixtures import GroupScene
from pointwise_rgbd.config import load_config
from pointwise_rgbd.group_process import load_group_noise
from pointwise_rgbd.surface_supernodes import build_partition
from pointwise_rgbd.group_sensor_tracks import independent_group_sensor_rows
from pointwise_rgbd.group_partition_epochs import run_partition_epochs
from pointwise_rgbd.group_coherence import select_moving_component,edge_strain
from rgbd_common.backends.torch_cuda import Backend


class GroupCoherenceTests(unittest.TestCase):
    def test_static_and_single_global_drift_never_become_articulation(self):
        c=load_config();b=Backend('cpu')
        for motion in ('static','translation'):
            s=GroupScene(frames=8,motion=motion);_,p=build_partition(s,'first',c,100)
            rows=independent_group_sensor_rows(p,None,s,c,b)
            result=select_moving_component(p,rows)
            self.assertEqual(result['status'],'NO_COHERENT_MOVING_COMPONENT');self.assertNotIn('MOVING_RIGID',result['labels'])
            if motion=='static':self.assertIn('STATIC_COMPATIBLE',result['labels'])

    def test_one_actual_translating_part_and_static_support_from_raw_sensor_data(self):
        c=load_config();b=Backend('cpu');s=GroupScene(frames=10,motion='boundary')
        with tempfile.TemporaryDirectory() as root:
            p,_,_=run_partition_epochs(s,c,load_group_noise(),b,root,target_occupancy=1000)
            rows=independent_group_sensor_rows(p,None,s,c,b);result=select_moving_component(p,rows)
            self.assertEqual(result['status'],'SUPPORTED_MOVING_CORE',str(result.get('candidates')))
            self.assertIn('STATIC_COMPATIBLE',result['labels']);self.assertIn('MOVING_RIGID',result['labels'])
            truth=p.source[:,2]%96>=44
            self.assertTrue(truth[result['point_support']].all())
            for report in result['candidates']:
                if 'same_raw_ids' in report:self.assertTrue(report['same_raw_ids']);self.assertEqual(report['heldout_raw_overlap_with_training'],0)


if __name__=='__main__':unittest.main()
