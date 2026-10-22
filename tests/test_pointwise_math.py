import unittest
from unittest.mock import patch
import numpy as np
from numpy.testing import assert_allclose,assert_array_equal
import torch
from pointwise_rgbd.config import load_config
from pointwise_rgbd.process6 import transition,process_covariance,predict
from pointwise_rgbd.projection import project,backproject
from pointwise_rgbd.filter_iekf import correct
from rgbd_common.projection_cuda import Camera


def camera():
    a=.3;r=np.array([[np.cos(a),0,np.sin(a)],[0,1,0],[-np.sin(a),0,np.cos(a)]])
    t=np.eye(4);t[:3,:3]=r;t[:3,3]=[.03,-.04,.1]
    return Camera('test',128,96,np.array([90.,100.,63.5,47.5]),t,0,np.inf)


class PointwiseMathTests(unittest.TestCase):
    def test_projection_backprojection_full_anisotropic_covariance_and_jacobian(self):
        cam=camera();s=torch.tensor([[.03,.04,1.2,.2,-.1,.5]],dtype=torch.float64)
        h,j=project(s,cam);eps=1e-6
        finite=np.column_stack([((project(s+torch.eye(6,dtype=s.dtype)[i]*eps,cam)[0]-project(s-torch.eye(6,dtype=s.dtype)[i]*eps,cam)[0])/(2*eps)).numpy()[0] for i in range(6)])
        assert_allclose(j.numpy()[0],finite,atol=1e-8);assert_array_equal(j.numpy()[0,:,3:],0)
        r=np.array([[.2,.03,.001],[.03,.3,0],[.001,0,.0002]])
        xyz,cov=backproject(h.numpy(),cam,r)
        assert_allclose(xyz,s.numpy()[:,:3],atol=1e-14)
        root=np.linalg.cholesky(cov[0]);self.assertGreater(np.linalg.norm(cov[0]-np.diag(np.diag(cov[0]))),0)
    def test_reverse_quadratic_factor_identity_anisotropic_and_nonuniform_steps(self):
        rng=np.random.default_rng(9)
        for dt in (.013,.04,.17):
            a=rng.normal(size=(3,3));qc=a@a.T+.1*np.eye(3);b=rng.normal(size=(3,3));qx=b@b.T*.01
            f=transition(dt);reverse=transition(dt,True);q=process_covariance(dt,qc,qx);qr=process_covariance(dt,qc,qx,True)
            assert_allclose(f@reverse,np.eye(6),atol=1e-16);assert_allclose(qr,reverse@q@reverse.T,atol=1e-16)
            self.assertGreater(np.linalg.eigvalsh(q).min(),0);self.assertGreater(np.linalg.eigvalsh(qr).min(),0)
            x=rng.normal(size=6);y=rng.normal(size=6);ef=y-f@x;er=x-reverse@y
            assert_allclose(ef@np.linalg.solve(q,ef),er@np.linalg.solve(qr,er),rtol=1e-12)
            self.assertTrue(np.all(np.diag(qr[:3,3:])<0))
    def test_linear_measurement_matches_KF_and_fixed_prior_not_recounted(self):
        c=load_config(tail_probability=.01);c.iterations=20;c.tolerance=1e-12
        state=torch.tensor([[.1,.2,.3,.4,.5,.6]],dtype=torch.float64);p=torch.eye(6,dtype=state.dtype)[None]*.2
        h=torch.tensor([[1.,0,.1,0,0,0],[0,1,0,0,0,0],[0,0,1,0,0,0]],dtype=state.dtype)
        z=torch.tensor([[.12,.25,1.]],dtype=state.dtype);r=torch.eye(3,dtype=state.dtype)[None]*.03
        def linear(x,_):return x@h.T,h.expand(len(x),-1,-1)
        matches=[{'point':torch.tensor([0]),'z':z,'R':r,'camera':object()}]
        with patch('pointwise_rgbd.filter_iekf.project',linear):s,post,stats=correct(state,p,matches,c)
        gain=p[0]@h.T@torch.linalg.inv(h@p[0]@h.T+r[0])
        assert_allclose(s.numpy()[0],(state[0]+gain@(z[0]-h@state[0])).numpy(),atol=1e-12)
        expected=p[0]-gain@h@p[0];assert_allclose(post.numpy()[0],expected.numpy(),atol=1e-12)
        eye=torch.eye(6,dtype=state.dtype);joseph=(eye-gain@h)@p[0]@(eye-gain@h).T+gain@r[0]@gain.T
        assert_allclose(post.numpy()[0],joseph.numpy(),atol=1e-12)
        for a,b in zip(stats['energies'],stats['energies'][1:]):self.assertTrue(bool((b<=a+1e-14).all()))
    def test_nonlinear_energy_and_prediction_only_carry_forward(self):
        c=load_config(tail_probability=.01);cam=camera();s=torch.tensor([[.01,.02,1.2,.1,-.03,.01]],dtype=torch.float64);p=torch.eye(6,dtype=s.dtype)[None]*.02
        z=project(s+torch.tensor([.02,.01,.01,0,0,0]),cam)[0]
        r=torch.tensor(c.measurement_covariance(np.array([float(z[0,2])])),dtype=s.dtype)
        result,cov,stats=correct(s,p,[{'point':torch.tensor([0]),'z':z,'R':r,'camera':cam}],c)
        self.assertLess(float(stats['after'][0]),float(stats['before'][0]));self.assertGreater(np.linalg.eigvalsh(cov.numpy()[0]).min(),0)
        initial=p.clone()
        for _ in range(4):
            pred,pp=predict(s,p,.04,c);s,p,report=correct(pred,pp,[],c)
            assert_array_equal(p.numpy(),pp.numpy());assert_array_equal(s.numpy(),pred.numpy())
        self.assertGreater(float(p[0,:3,:3].trace()),float(initial[0,:3,:3].trace()))
    def test_joint_multiview_NIS_matches_dense_cross_camera_innovation(self):
        c=load_config();s=torch.tensor([[.01,.02,1.2,.1,0,0]],dtype=torch.float64);p=torch.eye(6,dtype=s.dtype)[None]*.03
        first=camera();second=camera();second.world_to_camera[0,3]+=.08
        matches=[];hs=[];errors=[];rs=[]
        for i,cam in enumerate((first,second)):
            h,j=project(s,cam);z=h+torch.tensor([[.2*(i+1),-.1,.003]],dtype=s.dtype);r=torch.diag(torch.tensor([.2,.2,.0001],dtype=s.dtype))
            matches.append({'point':torch.tensor([0]),'z':z,'R':r[None],'camera':cam});hs.append(j[0]);errors.append(z[0]-h[0]);rs.append(r)
        _,_,stats=correct(s,p,matches,c);h=torch.cat(hs);error=torch.cat(errors);r=torch.block_diag(*rs)
        innovation=h@p[0]@h.T+r;expected=error@torch.linalg.solve(innovation,error)
        assert_allclose(float(stats['nis'][0]),float(expected),atol=1e-12)
    def test_invalid_noise_and_missing_tail_are_targeted(self):
        c=load_config();c.association_tail_probability=None
        with self.assertRaisesRegex(ValueError,'MISSING_ASSOCIATION_TAIL'):c.gate()
        with self.assertRaises(ValueError):transition(0)


if __name__=='__main__':unittest.main()
