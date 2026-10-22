import unittest
import numpy as np
from numpy.testing import assert_allclose
from rgbd_common.dataloader import camera_from_metadata
from rgbd_common.backends.torch_cuda import Backend,numpy
from rgbd_common.projection_cuda import transport_and_project
from rgbd_common.fusion import backproject
from parallel_fixtures import Scene,configuration


class CameraTests(unittest.TestCase):
    def test_optical_world_round_trip_and_metric_depth(self):
        s=Scene();c=s.cameras[0];c.world_to_camera[:3,3]=[.1,-.2,.3]
        b=Backend('cpu','float64');xyz,_,_,pixels=backproject(s.observation(0,0),configuration(),b)
        result=transport_and_project(b.tensor(xyz),b.tensor(np.zeros(len(xyz)),__import__('torch').long),b.tensor(np.eye(4)[None]),c)
        assert_allclose(numpy(result['uv']),np.c_[pixels%8,pixels//8],atol=1e-14)
        assert_allclose(numpy(result['depth']),1,atol=1e-14)
    def test_verified_optical_extrinsic_convention(self):
        t=np.eye(4);t[:3,3]=[1,2,3]
        c=camera_from_metadata(dict(name='anything',width=8,height=8,fx=10,fy=10,cx=3.5,cy=3.5,pose_convention='opencv_optical_camera_to_world',pose_matrix_4x4=t.tolist()))
        assert_allclose(c.world_to_camera,np.linalg.inv(t))
