"""V2 M3 regression tests: only nominal model depth and foreground evidence."""
import unittest
import numpy as np
import torch
from tests.grouped_fixtures import GroupScene
from pointwise_rgbd.config import load_config
from pointwise_rgbd.observations import samples
from pointwise_rgbd.point_visibility import project_bins
from pointwise_rgbd.association import associate
from pointwise_rgbd.free_phase1 import ForegroundConstraint
from pointwise_rgbd.foreground_index import ForegroundIndex
from rgbd_common.backends.torch_cuda import Backend


class VisibilityTests(unittest.TestCase):
    def verify(self,device):
        scene=GroupScene();frame=scene.observation(0,0);cfg=load_config();b=Backend(device)
        raw=samples(frame,cfg);states=np.c_[raw.xyz,np.zeros_like(raw.xyz)]
        # Identical rays with different nominal model depths: only the nearer
        # point is a first hit. Its ID wins a same-depth tie deterministically.
        duplicate=states[:1].copy();duplicate[:,:3]*=2
        cloud=np.concatenate([states,duplicate,states[:1]])
        ids=np.arange(len(cloud));_,bins=project_bins(b.tensor(cloud),frame.camera,b,b.tensor(ids,torch.long))
        visible=bins['visible_ids'].cpu().numpy()
        self.assertEqual(len(visible),len(states));self.assertNotIn(len(states),visible);self.assertNotIn(len(states)+1,visible)
        covariance=np.tile(np.eye(6)*.001,(len(cloud),1,1));rgb=np.concatenate([raw.rgb,raw.rgb[:1],raw.rgb[:1]])
        before,_=associate(cloud,covariance,rgb,ids,frame,cfg,b,np.full(frame.depth.size,-1,np.int64))
        # Temporal association uses measured foreground depth, not nominal model-bin culling.
        self.assertTrue(frame.valid.ravel()[before['pixel']].all())
        frame.depth[~frame.mask]=.0001;frame.rgb[~frame.mask]=[1.,0.,1.]
        after,_=associate(cloud,covariance,rgb,ids,frame,cfg,b,np.full(frame.depth.size,-1,np.int64))
        np.testing.assert_array_equal(before['pixel'],after['pixel']);np.testing.assert_array_equal(before['point'].cpu(),after['point'].cpu())
        np.testing.assert_array_equal(before['z'].cpu(),after['z'].cpu());np.testing.assert_array_equal(before['R'].cpu(),after['R'].cpu())
        # Equal-cost independent matches may be emitted in a different order
        # on CPU/CUDA. Compare the actual point/pixel correspondences, not the
        # incidental ordering of the returned factor rows.
        correspondence=np.c_[before['point'].cpu().numpy(),before['pixel']]
        return visible,correspondence[np.argsort(correspondence[:,0])]

    def test_nominal_occlusion_and_background_invariance_cpu(self):self.verify('cpu')

    @unittest.skipUnless(torch.cuda.is_available(),'Host CUDA required')
    def test_nominal_occlusion_and_background_invariance_cuda(self):
        cpu=self.verify('cpu');cuda=self.verify('cuda')
        for a,b in zip(cpu,cuda):np.testing.assert_array_equal(a,b)

    def test_mask_cpu_cuda_agree_and_unknown_premise_is_undetermined(self):
        scene=GroupScene();frame=scene.observation(0,0);frame.object_only_no_external_occluders=True
        cfg=load_config();state=np.array([[-.475,-.355,1.,0,0,0],[-.95,-.71,2.,0,0,0]])
        for device in ['cpu']+(['cuda'] if torch.cuda.is_available() else []):
            b=Backend(device);constraint=ForegroundConstraint([frame],None,b.tensor([0,1],torch.long),2,cfg,b)
            np.testing.assert_array_equal(constraint(b.tensor(state)).cpu(),[False,True])
        self.assertFalse(ForegroundIndex(scene,cfg).feasible(0,state[:,:3],None))
        frame.object_only_no_external_occluders=None
        self.assertTrue(ForegroundIndex(scene,cfg).feasible(0,state[:,:3],None))


if __name__=='__main__':unittest.main()
