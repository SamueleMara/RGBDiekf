import unittest
import numpy as np
from numpy.testing import assert_allclose
from rgbd_common.lie import exp,State
from rgbd_common.backends.torch_cuda import Backend
from ukf.sigma_points import sigma_points
from ukf.dynamics import augmented_predict,nonlinear_transition
from ukf.measurement import woodbury
from ukf.filter import UKF
from parallel_fixtures import configuration,VectorState,LinearProblem


class UKFMathTests(unittest.TestCase):
    def test_tiny_but_real_cross_body_prior_is_not_declared_independent(self):
        config=configuration();backend=Backend('cpu');estimator=UKF(config,backend)
        state=State(np.tile(np.eye(4),(2,1,1)),np.zeros((2,6)))
        prior=np.eye(24)*1e-18;prior[0,12]=prior[12,0]=2e-19
        _,covariance,report=estimator.predict(state,prior,.04)
        self.assertEqual(report['sigma_count'],97)
        assert_allclose(covariance[0,12],prior[0,12],rtol=1e-10,atol=1e-30)

    def test_separable_prediction_equals_independent_groups_and_keeps_coupled_prior(self):
        from scipy.linalg import block_diag
        c=configuration();b=Backend('cpu');estimator=UKF(c,b)
        s=State(np.array([np.eye(4),exp(np.array([.1,.2,.1,.1,-.2,.3]))]),np.array([[.1,.2,0,.1,0,0],[0,0,.1,0,0,.1]]))
        p=np.eye(24)*.001
        state,cov,report=estimator.predict(s,p,.043)
        independent=[estimator.predict(State(s.poses[g:g+1],s.twists[g:g+1]),p[12*g:12*g+12,12*g:12*g+12],.043) for g in range(2)]
        assert_allclose(cov,block_diag(*[row[1] for row in independent]),atol=1e-12)
        self.assertEqual(report['sigma_count_per_group'],[49,49])
        p[0,12]=p[12,0]=.0002
        _,coupled,report=estimator.predict(s,p,.043)
        self.assertEqual(report['sigma_count'],97);self.assertGreater(abs(coupled[0,12]),.0001)
    def test_batched_conditional_sigma_reprojection_matches_scalar_full_rgbd_operator(self):
        from parallel_fixtures import Scene
        from rgbd_common.fusion import fuse_frame_zero
        from rgbd_common.matching_cuda import match_rgbd
        from rgbd_common.measurement_problem import MeasurementProblem
        from ukf.filter import GroupChart
        from ukf.measurement import measurement_moments
        from oracle.ukf_measurement_scalar import measurement_moments as scalar
        from rgbd_common.backends.torch_cuda import numpy
        c=configuration();b=Backend('cpu');scene=Scene(3,2);model,_=fuse_frame_zero(scene,c,b)
        model.group=np.arange(model.count,dtype=np.int32)%2
        state=State(np.array([np.eye(4),exp(np.array([.002,0,0,0,.003,0]))]),np.zeros((2,6)))
        matches=[match_rgbd(model,state,scene.observation(1,ci),c,b)[0] for ci in range(2)]
        # Include RGB-depth and pixel/color cross terms; removing constant
        # sigma RGB rows must never drop color residuals from R^{-1} e.
        rng=np.random.default_rng(73);root=rng.normal(size=(6,6));small=root@root.T+np.eye(6)
        for heterogeneous in (False,True):
            for match in matches:
                match.covariance=b.tensor(small).expand(len(match),-1,-1)
                if heterogeneous:match.covariance=match.covariance.clone()+b.tensor(np.linspace(0,.1,len(match)))[:,None,None]*b.tensor(np.eye(6))
            problem=MeasurementProblem(model,matches,b)
            for group in (None,0,1):
                chart=state if group is None else GroupChart(state,group)
                selected=problem if group is None else problem.for_group(group)
                for scale in (1e-7,.05):
                    p=np.eye(chart.dim)*scale
                    try:expected=scalar(selected,chart,p)
                    except ValueError as error:
                        self.assertIn('UKF_CONDITIONAL_PROJECTION_UNDEFINED',str(error))
                        with self.assertRaisesRegex(ValueError,'UKF_CONDITIONAL_PROJECTION_UNDEFINED'):measurement_moments(selected,chart,p)
                        continue
                    b.chunk_size=lambda _:5
                    actual=measurement_moments(selected,chart,p)
                    for x,y in zip(expected[:4],actual[:4]):
                        if isinstance(x,dict):
                            for key in ('active_factors','deferred_visibility_factors'):self.assertEqual(x[key],y[key])
                        else:assert_allclose(numpy(x) if hasattr(x,'detach') else x,numpy(y) if hasattr(y,'detach') else y,atol=1e-10,rtol=1e-12)

    def test_sigma_mean_covariance_and_positive_weights(self):
        rng=np.random.default_rng(1);root=rng.normal(size=(12,12));p=root@root.T
        eta,wm,wc=sigma_points(p)
        assert_allclose(wm@eta,0,atol=1e-14);assert_allclose(eta.T@(eta*wc[:,None]),p,atol=1e-12)
        self.assertTrue(np.all(wc>=0));self.assertEqual(len(eta),25)
    def test_augmented_prediction_49_and_zero_noise_constant_twist(self):
        state=State(exp(np.array([.1,.2,.3,.2,-.1,.3]))[None],np.array([[.1,.2,.3,.2,.1,-.1]]))
        expected=nonlinear_transition(state,.047,np.zeros(12))
        pred,p,cross,report=augmented_predict(state,np.zeros((12,12)),.047,np.zeros((12,12)))
        assert_allclose(pred.minus(expected),0,atol=1e-14);assert_allclose(p,0,atol=1e-28);assert_allclose(cross,0,atol=1e-28);self.assertEqual(report['sigma_count'],49)
    def test_Woodbury_actual_filter_vs_linear_Kalman_and_dense_innovation(self):
        rng=np.random.default_rng(4);h=rng.normal(size=(9,12));p=np.eye(12)*.2;r=np.diag(np.arange(1,10)*.07);z=rng.normal(size=9)
        b=Backend('cpu');n=configuration();problem=LinearProblem(h,r,z)
        def reducer(problem,base,pminus,parameters):
            eta,wm,wc=sigma_points(pminus,*parameters);a=eta.T*np.sqrt(wc)[None,:]
            y=eta@h.T;mean=wm@y;zz=(y-mean).T*np.sqrt(wc)[None,:]
            m=np.eye(len(eta))+zz.T@np.linalg.solve(r,zz);bb=zz.T@np.linalg.solve(r,z-mean)
            return a,b.tensor(m),b.tensor(bb),{'sigma_count':len(eta)}
        state,cov,report=UKF(n,b,reducer).correct(VectorState(np.zeros(12)),p,problem)
        innovation=h@p@h.T+r;gain=np.linalg.solve(innovation,h@p).T
        assert_allclose(state.value,gain@z,atol=1e-12);assert_allclose(cov,p-gain@innovation@gain.T,atol=1e-12)
        self.assertEqual(report['sigma_count'],25)
        self.assertGreater(report['proposal_inf'],.001)
        self.assertLess(report['inner_solve_metric'],1e-12)
        self.assertIn('no MAP-stationarity claim',report['inner_stationarity_kind'])
    def test_negative_covariance_not_hidden_by_jitter(self):
        bad=np.eye(12);bad[0,0]=-.01
        with self.assertRaises(ValueError):sigma_points(bad)
