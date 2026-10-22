import unittest
import numpy as np
from numpy.testing import assert_allclose
from rgbd_common.lie import exp,log,inverse,adjoint,State,inverse_transport


class LieTests(unittest.TestCase):
    def test_canonical_rigid_gauge_change_preserves_transport_and_outputs(self):
        from rgbd_common.backends.torch_cuda import Backend,numpy
        from rgbd_common.observation_model import output
        from parallel_fixtures import Scene
        import torch
        c=exp(np.array([.2,-.3,.1,.4,-.2,.1]));t=exp(np.array([-.1,.2,.3,.2,.1,-.3]));p=np.array([[.1,.2,1.]])
        transformed=p@c[:3,:3].T+c[:3,3];b=Backend('cpu');gid=b.tensor([0],torch.long);rgb=b.tensor([[.2,.3,.4]])
        original=output(State(t[None],np.zeros((1,6))),b.tensor(p),rgb,gid,Scene().cameras[0],b)[0]
        changed=output(State((t@inverse(c))[None],np.zeros((1,6))),b.tensor(transformed),rgb,gid,Scene().cameras[0],b)[0]
        assert_allclose(numpy(original),numpy(changed),atol=1e-13)
    def test_exp_log_composition_and_adjoint(self):
        x=np.array([.2,-.1,.3,.5,-.4,.2]);t=exp(x)
        assert_allclose(log(t),x,atol=1e-14);assert_allclose(t@inverse(t),np.eye(4),atol=1e-14)
        d=np.array([.1,.2,-.3,.4,.2,.1])*1e-6
        assert_allclose(t@exp(d)@inverse(t),exp(adjoint(t)@d),atol=1e-14)
    def test_same_relative_transport_in_both_time_directions(self):
        a=exp(np.array([.1,.2,.3,.2,-.1,.4]));b=exp(np.array([-.1,.4,.2,.3,.2,.1]));q=np.array([.2,.3,1.])
        target=inverse_transport(q,a,b);assert_allclose(inverse_transport(target,b,a),q,atol=1e-14)
