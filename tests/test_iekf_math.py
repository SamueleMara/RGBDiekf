import unittest
import numpy as np
from numpy.testing import assert_allclose
from rgbd_common.lie import exp,State
from rgbd_common.backends.torch_cuda import Backend,numpy
from iekf.dynamics import transition
from iekf.measurement import h_H
from iekf.filter import IEKF
from parallel_fixtures import Scene,configuration,VectorState,LinearProblem


class IEKFMathTests(unittest.TestCase):
    def test_analytic_F_G_nonzero_pose_twist_and_irregular_dt(self):
        state=State(exp(np.array([.1,-.2,.3,.2,-.1,.4]))[None],np.array([[.1,.2,-.1,.2,.3,-.2]]))
        from ukf.dynamics import nonlinear_transition
        for dt in (.013,.047,.11):
            pred,f,g,_=transition(state,dt,np.eye(12)*.001);eps=1e-6
            fd=np.column_stack([(transition(state.plus(np.eye(12)[j]*eps),dt,np.eye(12))[0].minus(pred)-transition(state.plus(-np.eye(12)[j]*eps),dt,np.eye(12))[0].minus(pred))/(2*eps) for j in range(12)])
            gd=np.column_stack([(nonlinear_transition(state,dt,np.eye(12)[j]*eps).minus(pred)-nonlinear_transition(state,dt,-np.eye(12)[j]*eps).minus(pred))/(2*eps) for j in range(12)])
            assert_allclose(f,fd,atol=1e-9);assert_allclose(g,gd,atol=1e-9)
    def test_conditional_six_dimensional_H(self):
        import torch
        b=Backend('cpu');s=Scene();camera=s.cameras[0]
        state=State(exp(np.array([.01,.02,.03,.12,-.07,.03]))[None],np.ones((1,6))*.1)
        xyz=b.tensor([[.02,.03,1.1]]);rgb=b.tensor([[.2,.3,.4]]);group=b.tensor([0],torch.long)
        _,h=h_H(state,xyz,rgb,group,camera,b);eps=1e-6
        fd=np.column_stack([(numpy(h_H(state.plus(np.eye(12)[j]*eps),xyz,rgb,group,camera,b)[0])[0]-numpy(h_H(state.plus(-np.eye(12)[j]*eps),xyz,rgb,group,camera,b)[0])[0])/(2*eps) for j in range(12)])
        assert_allclose(numpy(h)[0],fd,atol=1e-8);assert_allclose(numpy(h)[0,3:],0)
    def test_compact_pose_reduction_equals_full_12_column_reference(self):
        import torch
        from rgbd_common.fusion import fuse_frame_zero
        from rgbd_common.matching_cuda import match_rgbd
        from rgbd_common.measurement_problem import MeasurementProblem
        from iekf.measurement import reduce_information
        config=configuration();backend=Backend('cpu');scene=Scene(3,1);model,_=fuse_frame_zero(scene,config,backend)
        model.group=np.arange(model.count,dtype=np.int32)%2
        state=State(np.tile(np.eye(4),(2,1,1)),np.zeros((2,6)))
        matches,*_=match_rgbd(model,state,scene.observation(1,0),config,backend)
        rng=np.random.default_rng(58);root=rng.normal(size=(6,6));r=root@root.T+np.eye(6)
        matches.covariance=backend.tensor(r).expand(len(matches),-1,-1)
        problem=MeasurementProblem(model,[matches],backend);a,b=reduce_information(problem,state)
        predicted,h=h_H(state,problem.xyz[matches.point],problem.rgb[matches.point],problem.group[matches.point],matches.camera,backend)
        jac=numpy(h);error=numpy(matches.observed-predicted);labels=numpy(problem.group[matches.point])
        for group in range(2):
            ids=np.flatnonzero(labels==group);hh=jac[ids].reshape(-1,12)
            expected_a=sum((row.T@np.linalg.solve(r,row) for row in jac[ids]),np.zeros((12,12)))
            expected_b=sum((row.T@np.linalg.solve(r,e) for row,e in zip(jac[ids],error[ids])),np.zeros(12))
            assert_allclose(numpy(a[group]),expected_a,atol=1e-12);assert_allclose(numpy(b[group]),expected_b,atol=1e-12)
            assert_allclose(numpy(a[group])[6:],0);assert_allclose(numpy(b[group])[6:],0)

    def test_information_stationarity_is_invariant_to_coordinate_units(self):
        from iekf.filter import newton_decrement
        rng=np.random.default_rng(13);root=rng.normal(size=(12,12));information=root@root.T+np.eye(12);gradient=rng.normal(size=12)
        scale=np.diag([1000.,1000.,1000.,1.,1.,1.,1000.,1000.,1000.,1.,1.,1.]);inverse=np.linalg.inv(scale)
        assert_allclose(newton_decrement(gradient,information),newton_decrement(inverse.T@gradient,inverse.T@information@inverse),atol=1e-12)

    def test_actual_filter_linear_gaussian_Joseph_fixed_prior_and_damping(self):
        rng=np.random.default_rng(12);h=rng.normal(size=(8,12));r=np.diag(np.arange(1,9)*.1);p=np.eye(12)*.3;z=rng.normal(size=8)
        problem=LinearProblem(h,r,z);b=Backend('cpu');n=configuration()
        def reducer(problem,state):
            a=h.T@np.linalg.solve(r,h);g=h.T@np.linalg.solve(r,z-h@state.value)
            return b.tensor(a[None]),b.tensor(g[None])
        expected_p=np.linalg.inv(np.linalg.inv(p)+h.T@np.linalg.solve(r,h));expected=expected_p@h.T@np.linalg.solve(r,z)
        outputs=[]
        for iterations,damping in ((2,1e-6),(12,1e-6),(12,.01)):
            n.numerical.update(inner_iterations=iterations,damping=damping,gradient_tolerance=1e-9)
            state,cov,report=IEKF(n,b,reducer).correct(VectorState(np.zeros(12)),p,problem)
            assert_allclose(state.value,expected,atol=1e-8);assert_allclose(cov,expected_p,atol=1e-12);outputs.append(cov)
        gain=np.linalg.solve(h@p@h.T+r,h@p).T;eye=np.eye(12)-gain@h
        assert_allclose(outputs[0],eye@p@eye.T+gain@r@gain.T,atol=1e-12)
