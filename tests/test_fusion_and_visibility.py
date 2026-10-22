import unittest
import numpy as np
from numpy.testing import assert_allclose
from rgbd_common.backends.torch_cuda import Backend,numpy
from rgbd_common.fusion import fuse_frame_zero,backproject,fuse,source_latent_covariance
from rgbd_common.geometry import ObservationIDs
from rgbd_common.projection_cuda import transport_and_project,build_pixel_bins_and_zbuffer
from rgbd_common.lie import State,exp
from parallel_fixtures import Scene,configuration
import torch


class FusionTests(unittest.TestCase):
    def test_point_ids_are_never_recycled_after_merge_or_checkpoint(self):
        import tempfile
        from rgbd_common.geometry import PointMap
        s=Scene();c=configuration();b=Backend('cpu');m,_=fuse_frame_zero(s,c,b);last=int(m.point_id[-1])
        compact=m.take(np.arange(m.count-1))
        with tempfile.TemporaryDirectory() as root:
            compact.save(root+'/points.npz');restored=PointMap.load(root+'/points.npz').copy()
            restored.append(m.xyz[-1:],m.rgb[-1:],[0],m.covariance[-1:],m.source_id[-1:])
            self.assertGreater(restored.point_id[-1],last)
    def test_exported_source_handles_share_latents_without_dense_point_covariance(self):
        import tempfile,json
        from pathlib import Path
        from rgbd_common.source_factors import export_source_factors
        s=Scene();b=Backend('cpu');c=configuration();m,_=fuse_frame_zero(s,c,b)
        states=[State(np.eye(4)[None],np.zeros((1,6))) for _ in s.times]
        with tempfile.TemporaryDirectory() as root:
            report=export_source_factors(m,states,s,c,b,root)
            self.assertEqual(report['factors'],128)
            one=np.load(Path(root)/'source_000000_000.npz');two=np.load(Path(root)/'source_000000_001.npz')
            assert_allclose(one['source_time'],two['source_time']);assert_allclose(one['source_group'],two['source_group'])
            self.assertTrue(set(one['raw_id']).isdisjoint(two['raw_id']))
            # Export is the relinearized source Gaussian with the ORIGINAL
            # proper geometry prior, not the two-sample initializer alone.
            _,_,sensor,_=backproject(s.observation(0,0),c,b)
            precision=np.linalg.inv(sensor)
            expected=np.linalg.inv(np.linalg.inv(c.map_covariance)+2*precision)@precision
            assert_allclose(one['fusion_weight'],expected,atol=1e-13)
            self.assertEqual(one['point_pose_jacobian'].shape,(64,6,6))
    def test_same_multicamera_points_fuse_with_two_sources(self):
        s=Scene();b=Backend('cpu');m,records=fuse_frame_zero(s,configuration(),b)
        self.assertEqual(m.count,64);self.assertEqual(records[1]['fused_existing'],64)
        self.assertTrue(all(len(set(row))==2 for row in m.provenance))
        self.assertFalse(hasattr(m,'normals'));self.assertFalse(hasattr(m,'edges'))
    def test_replayed_source_does_not_shrink_covariance(self):
        s=Scene(2,1);b=Backend('cpu');n=configuration();m,_=fuse_frame_zero(s,n,b)
        x,color,cov,pix=backproject(s.observation(0,0),n,b);before=m.covariance.copy()
        count=fuse(m,np.arange(64),x,color,cov,ObservationIDs(s).encode(0,0,pix))
        self.assertEqual(count,0);assert_allclose(m.covariance,before)
    def test_depth_bin_hides_deeper_color_and_ties_are_deterministic(self):
        s=Scene();b=Backend('cpu','float64');camera=s.cameras[0]
        xyz=b.tensor([[-.05,-.05,1],[-.1,-.1,2],[-.05,-.05,1]])
        p=transport_and_project(xyz,b.tensor([0,0,0],torch.long),b.tensor(np.eye(4)[None]),camera)
        for _ in range(3):
            bins=build_pixel_bins_and_zbuffer(p,camera);self.assertEqual(numpy(bins['visible_ids']).tolist(),[0])
    def test_shared_source_pose_creates_cross_covariance(self):
        pose=exp(np.array([.1,.2,.3,.2,.1,-.3]));ps=np.eye(6)*.01;ry=np.eye(3)*.0001
        pi,ci,ji=source_latent_covariance(np.array([.1,.2,1.]),pose,ry,ps,np.zeros((3,6)))
        pj,cj,jj=source_latent_covariance(np.array([.3,.2,1.]),pose,ry,ps,np.zeros((3,6)))
        self.assertGreater(np.linalg.norm(ji@ps@jj.T),0)
        assert_allclose(ci,ji@ps);self.assertGreater(np.linalg.eigvalsh(pi).min(),0)
