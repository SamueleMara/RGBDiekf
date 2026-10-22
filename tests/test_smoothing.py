import unittest
import numpy as np
from numpy.testing import assert_allclose
from rgbd_common.estimator import Step
from iekf.smoother import smooth as analytical_smooth
from ukf.smoother import smooth as unscented_smooth
from parallel_fixtures import VectorState


class SmoothingTests(unittest.TestCase):
    def test_independent_smoothers_vs_dense_linear_batch(self):
        dim=12;frames=4;f=np.eye(dim);f[:6,6:]=np.eye(6)*.04
        p0=np.eye(dim);q=np.eye(dim)*.03;r=np.eye(dim)*.2
        z=np.array([np.arange(dim)*.01+k*.1 for k in range(frames)])
        mean=np.zeros(dim);p=p0.copy();steps=[]
        for k in range(frames):
            pred=mean if k==0 else f@mean;pm=p if k==0 else f@p@f.T+q
            cross=None if k==0 else p@f.T
            gain=np.linalg.solve(pm+r,pm).T;mean=pred+gain@(z[k]-pred);p=pm-gain@pm
            steps.append(Step(VectorState(pred),pm,VectorState(mean),p,None if k==0 else f,cross))
        precision=np.zeros((frames*dim,frames*dim));rhs=np.zeros(frames*dim);precision[:dim,:dim]+=np.linalg.inv(p0)
        for k in range(frames):
            idx=slice(k*dim,(k+1)*dim);precision[idx,idx]+=np.linalg.inv(r);rhs[idx]+=np.linalg.solve(r,z[k])
            if k:
                prev=slice((k-1)*dim,k*dim);w=np.linalg.inv(q)
                precision[prev,prev]+=f.T@w@f;precision[idx,idx]+=w;precision[prev,idx]-=f.T@w;precision[idx,prev]-=w@f
        covariance=np.linalg.inv(precision);expected=covariance@rhs
        for solver in (analytical_smooth,unscented_smooth):
            states,cov,report=solver(steps)
            from rgbd_common.trajectory_covariance import TrajectoryMoments
            moments=TrajectoryMoments(steps,(states,cov,report))
            for k in range(frames):
                assert_allclose(states[k].value,expected[k*dim:(k+1)*dim],atol=1e-12)
                assert_allclose(cov[k],covariance[k*dim:(k+1)*dim,k*dim:(k+1)*dim],atol=1e-12)
                for t in range(frames):assert_allclose(moments.block(k,t),covariance[k*dim:(k+1)*dim,t*dim:(t+1)*dim],atol=1e-12)
            self.assertGreater(np.linalg.norm(states[0].value-steps[0].filtered.value),0)
    def test_gaussian_references_keep_nonzero_mean_and_correct_chart_covariance(self):
        from rgbd_common.trajectory_covariance import error_moments_at_reference
        from rgbd_common.lie import State,exp
        reference=State(exp(np.array([.1,.2,-.1,.2,-.1,.3]))[None],np.zeros((1,6)))
        mean=reference.plus(np.array([.2,-.1,.03,.1,.2,-.15,.1,.2,0,0,.1,0]));p=np.eye(12)*.001
        offset,cov,second=error_moments_at_reference([mean],[p],[reference]);eps=1e-6
        derivative=np.column_stack([(mean.plus(np.eye(12)[j]*eps).minus(reference)-mean.plus(-np.eye(12)[j]*eps).minus(reference))/(2*eps) for j in range(12)])
        assert_allclose(cov[0],derivative@p@derivative.T,atol=1e-11)
        assert_allclose(offset[0],mean.minus(reference),atol=1e-14)
        assert_allclose(second[0],cov[0]+np.outer(offset[0],offset[0]),atol=1e-14)
        self.assertGreater(np.linalg.norm(offset),.1)

    def test_unscented_smoother_refuses_missing_sigma_cross_covariance(self):
        states=[Step(VectorState(np.zeros(12)),np.eye(12),VectorState(np.zeros(12)),np.eye(12)) for _ in range(2)]
        with self.assertRaises(ValueError):unscented_smooth(states)
