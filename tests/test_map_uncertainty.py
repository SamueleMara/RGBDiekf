import tempfile,unittest
from pathlib import Path
import numpy as np
from numpy.testing import assert_allclose
import torch
from rgbd_common.backends.torch_cuda import Backend,numpy
from rgbd_common.lie import State,exp
from rgbd_common.fusion import fuse_frame_zero
from rgbd_common.map_uncertainty import point_output_jacobian,conditional_map_uncertainty
from rgbd_common.observation_model import covariance
from parallel_fixtures import Scene,configuration


class MapUncertaintyTests(unittest.TestCase):
    def test_raw_point_jacobian_vs_finite_difference_with_nonzero_pose(self):
        scene=Scene();backend=Backend('cpu');state=State(exp(np.array([.02,.01,.03,.04,.05,-.03]))[None],np.zeros((1,6)))
        point=backend.tensor([[.02,.03,1.1]]);color=backend.tensor([[.2,.3,.4]]);group=backend.tensor([0],torch.long)
        value,h=point_output_jacobian(state,point,color,group,scene.cameras[0],backend);eps=1e-6
        def prediction(x):return numpy(point_output_jacobian(state,backend.tensor(x[None,:3]),backend.tensor(x[None,3:]),group,scene.cameras[0],backend)[0])[0]
        x=np.r_[numpy(point)[0],numpy(color)[0]]
        finite=np.column_stack([(prediction(x+np.eye(6)[j]*eps)-prediction(x-np.eye(6)[j]*eps))/(2*eps) for j in range(6)])
        assert_allclose(numpy(h)[0],finite,atol=1e-8)
    def test_selected_pose_point_cross_information_vs_full_tiny_factor(self):
        from rgbd_common.map_uncertainty import pose_point_information
        scene=Scene();b=Backend('cpu');state=State(exp(np.array([.02,.01,.03,.04,.05,-.03]))[None],np.zeros((1,6)))
        point=b.tensor([[.02,.03,1.1]]);color=b.tensor([[.2,.3,.4]]);group=b.tensor([0],torch.long)
        rng=np.random.default_rng(61);root=rng.normal(size=(6,6));r=root@root.T+np.eye(6)
        blocks=pose_point_information(state,point,color,group,scene.cameras[0],b.tensor(r),b)
        _,hp=point_output_jacobian(state,point,color,group,scene.cameras[0],b);eps=1e-6
        ht=np.column_stack([(numpy(point_output_jacobian(state.plus(np.r_[np.eye(6)[j]*eps,np.zeros(6)]),point,color,group,scene.cameras[0],b)[0])[0]-numpy(point_output_jacobian(state.plus(np.r_[-np.eye(6)[j]*eps,np.zeros(6)]),point,color,group,scene.cameras[0],b)[0])[0])/(2*eps) for j in range(6)])
        h=np.c_[numpy(hp)[0],ht];expected=h.T@np.linalg.solve(r,h)
        for actual,target in zip(blocks,(expected[:6,:6],expected[:6,6:],expected[6:,6:])):assert_allclose(numpy(actual)[0],target,atol=1e-7)

    def test_all_actual_owners_and_original_prior_once_with_unseen_point(self):
        scene=Scene(3,2);backend=Backend('cpu');config=configuration();model,_=fuse_frame_zero(scene,config,backend)
        # A genuinely unobserved, outside-image hypothesis retains the proper
        # original prior; it must not inherit a tiny source-fusion covariance.
        model.append([[100,100,1]],[[.2,.3,.4]],[0],np.eye(6)[None]*.0001,[128])
        states=[State(np.eye(4)[None],np.zeros((1,6))) for _ in scene.times]
        with tempfile.TemporaryDirectory() as root:
            result=conditional_map_uncertainty(model,states,scene,config,backend,root)
            with np.load(Path(root)/'conditional_map_covariance.npz') as data:
                assert_allclose(data['conditional_covariance'][-1],config.map_covariance,atol=1e-12)
                self.assertEqual(data['owned_raw_factor_count'][-1],0)
                _,h=point_output_jacobian(states[0],backend.tensor(model.xyz[:1]),backend.tensor(model.rgb[:1]),backend.tensor([0],torch.long),scene.cameras[0],backend)
                r=numpy(covariance(config,backend.tensor([1.]),backend))[0];j=numpy(h)[0]
                expected=np.linalg.inv(np.linalg.inv(config.map_covariance)+6*j.T@np.linalg.solve(r,j))
                assert_allclose(data['conditional_covariance'][0],expected,atol=1e-12)
                self.assertEqual(data['owned_raw_factor_count'][0],6)
            self.assertEqual(result['owned_raw_factors'],64*6);self.assertEqual(result['unobserved_points'],1)
