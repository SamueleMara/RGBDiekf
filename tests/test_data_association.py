import unittest
import numpy as np
from numpy.testing import assert_allclose
from rgbd_common.backends.torch_cuda import Backend,numpy
from rgbd_common.fusion import fuse_frame_zero
from rgbd_common.lie import State
from rgbd_common.matching_cuda import match_rgbd
from rgbd_common.geometry import RawObservationOwnership
from parallel_fixtures import Scene,configuration


class AssociationTests(unittest.TestCase):
    def test_unique_raw_ownership_and_mask_domain(self):
        s=Scene();b=Backend('cpu');n=configuration();m,_=fuse_frame_zero(s,n,b)
        frame=s.observation(1,0);frame.mask[:4]=False
        matches,stats,_,_=match_rgbd(m,State(np.eye(4)[None],np.zeros((1,6))),frame,n,b)
        pixels=numpy(matches.pixel)
        self.assertEqual(len(pixels),len(np.unique(pixels)));self.assertTrue(np.all(pixels>=32))
        self.assertGreater(stats['matched'],0)
    def test_no_return_is_not_free_space_or_rgb_evidence(self):
        s=Scene();b=Backend('cpu');n=configuration();m,_=fuse_frame_zero(s,n,b)
        frame=s.observation(1,0);frame.depth[:]=0;frame.rgb[:]=[1,0,0]
        matches,stats,_,_=match_rgbd(m,State(np.eye(4)[None],np.zeros((1,6))),frame,n,b)
        self.assertEqual(len(matches),0);self.assertEqual(stats['free_space_rejected'],0)
    def test_wrong_depth_reduces_match_likelihood_and_leaves_raw_unsolved(self):
        s=Scene();b=Backend('cpu');n=configuration();m,_=fuse_frame_zero(s,n,b)
        frame=s.observation(1,0);state=State(np.eye(4)[None],np.zeros((1,6)))
        good=match_rgbd(m,state,frame,n,b)[0]
        m.xyz*=.5;bad,stats,_,_=match_rgbd(m,state,frame,n,b)
        self.assertGreater(len(good),len(bad));self.assertGreater(stats['free_space_rejected'],0)
    def test_raw_ownership_rejects_double_count_and_refreshes(self):
        import tempfile
        s=Scene()
        with tempfile.TemporaryDirectory() as root:
            owner=RawObservationOwnership(s,root);owner.write(1,0,[0,1],[3,4])
            self.assertEqual(len(owner.unsolved(1,0)),62)
            with self.assertRaises(ValueError):owner.write(1,0,[1,1],[3,4])
            owner.write(1,0,[],[]);self.assertEqual(len(owner.unsolved(1,0)),64)
