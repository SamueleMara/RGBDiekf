import tempfile,unittest
from pathlib import Path
import numpy as np
from numpy.testing import assert_array_equal
from scipy.sparse.csgraph import connected_components
from pointwise_fixtures import PointScene,config
from pointwise_rgbd.surface_supernodes import build_partition,surface_graph,connected_geodesic_assignment,endpoint_samples


class GroupPartitionTests(unittest.TestCase):
    def test_endpoint_direction_all_raw_sources_and_surface_connectivity(self):
        scene=PointScene(frames=5,cameras=2,hidden=True);c=config()
        for anchor,frame in [('first',0),('last',4)]:
            k,p=build_partition(scene,anchor,c,target_occupancy=64)
            self.assertEqual(k,frame);self.assertEqual(len(p.xyz),sum(scene.observation(frame,ci).valid.sum() for ci in range(2)))
            assert_array_equal(p.source[:,0],scene.frame_ids[frame])
            again=build_partition(scene,anchor,c,target_occupancy=64)[1];self.assertEqual(p.digest(),again.digest())
            with tempfile.TemporaryDirectory() as root:
                path=Path(root)/'partition.npz';p.save(path)
                with np.load(path) as data:assert_array_equal(data['membership'],p.membership)
    def test_close_depth_layers_not_joined_and_isolated_samples_retained(self):
        scene=PointScene(frames=3,cameras=1,hidden=False)
        frame=scene.observation(0,0);frame.mask[:]=False;frame.depth[:]=0
        for x,d in [(30,1.),(31,1.),(32,1.2),(33,1.2),(50,2.)]:frame.mask[24,x]=True;frame.depth[24,x]=d;frame.rgb[24,x]=[.8,.1,.2]
        k,p=build_partition(scene,'first',config(),target_occupancy=64)
        self.assertEqual(len(p.xyz),5);self.assertEqual(p.count,3)
        self.assertNotEqual(p.membership[1],p.membership[2]);self.assertEqual(p.membership[0],p.membership[1])
    def test_geodesic_seeds_connected_and_no_raw_sample_dropped(self):
        scene=PointScene(frames=3,cameras=1,hidden=False);frame=scene.observation(0,0);frame.mask[:]=False;frame.depth[:]=0
        frame.mask[20:28,28:36]=True;frame.depth[frame.mask]=1.;frame.rgb[frame.mask]=[.8,.2,.1]
        k,p=build_partition(scene,'first',config(),target_occupancy=8)
        self.assertEqual(len(p.xyz),64);self.assertEqual(p.count,8)
        _,raw,xyz,_,cov,_=endpoint_samples(scene,'first',config());_,graph,_=surface_graph(raw,xyz,cov,config())
        for g in range(p.count):
            indices=np.flatnonzero(p.membership==g)
            self.assertEqual(connected_components(graph[indices][:,indices],directed=False,return_labels=False),1)


if __name__=='__main__':unittest.main()
