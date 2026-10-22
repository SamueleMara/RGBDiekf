import unittest
import numpy as np
from numpy.testing import assert_allclose
from rgbd_common.fusion import source_latent_covariance
from rgbd_common.lie import exp
from iekf.covariance import selected_covariance
from scipy.sparse.linalg import aslinearoperator


class CovarianceTests(unittest.TestCase):
    def test_shared_covariance_precision_is_exact_and_depth_law_stays_variable(self):
        import torch
        from parallel_fixtures import configuration
        from rgbd_common.backends.torch_cuda import Backend,numpy
        from rgbd_common.observation_model import apply_precision,covariance,gaussian_log_probability,select_covariance
        b=Backend('cpu');c=configuration();rng=np.random.default_rng(78)
        root=rng.normal(size=(6,6));small=b.tensor(root@root.T+np.eye(6));rhs=b.tensor(rng.normal(size=(17,6,12)))
        shared=small.expand(17,-1,-1)
        assert_allclose(numpy(apply_precision(shared,rhs)),numpy(torch.linalg.solve(shared.clone(),rhs)),atol=1e-12)
        empty=small.expand(0,-1,-1);self.assertEqual(select_covariance(empty,b.tensor([],torch.bool)).shape,(0,6,6))
        c.rgbd_depth_squared=np.eye(4)*.00001;depth=b.tensor(np.linspace(.1,2,17));r=covariance(c,depth,b)
        self.assertNotEqual(r.stride(0),0);self.assertFalse(torch.equal(r[0],r[-1]))
        e=rhs[:,:,0];reference=-.5*((e*torch.linalg.solve(r,e[...,None])[...,0]).sum(1)+torch.linalg.slogdet(r).logabsdet+6*np.log(2*np.pi))
        assert_allclose(numpy(gaussian_log_probability(e,r)),numpy(reference),atol=1e-10)
    def test_production_inner_problem_excludes_initialization_self_fit(self):
        from parallel_fixtures import Scene,configuration
        from rgbd_common.geometry import ObservationIDs
        from rgbd_common.fusion import fuse_frame_zero
        from rgbd_common.matching_cuda import match_rgbd
        from rgbd_common.measurement_problem import MeasurementProblem
        from rgbd_common.backends.torch_cuda import Backend
        from rgbd_common.lie import State
        s=Scene(3,1);c=configuration();b=Backend('cpu');m,_=fuse_frame_zero(s,c,b);state=State(np.eye(4)[None],np.zeros((1,6)))
        for k in (0,1):
            match,*_=match_rgbd(m,state,s.observation(k,0),c,b)
            p=MeasurementProblem(m,[match],b,ObservationIDs(s))
            self.assertEqual(sum(map(len,p.matches)),0 if k==0 else 64)
            self.assertEqual(p.excluded_source_factors,64 if k==0 else 0)
    def test_shared_pose_augmentation_monte_carlo_and_self_source_cancellation(self):
        from rgbd_common.source_factors import source_jacobians,augmentation_blocks
        rng=np.random.default_rng(91);pose=exp(np.array([.1,.2,-.1,.2,-.1,.3]));world=np.array([.1,.2,1.])
        jy,jt=source_jacobians(world,pose);other=source_jacobians(world+[.1,.1,0],pose)[1]
        raw=np.diag([.001,.002,.003,.001,.001,.001]);p=np.eye(6)*.0001
        joint=np.zeros((12,12));joint[:6,:6]=raw;joint[6:,6:]=p
        pp,pt=augmentation_blocks(jy,jt,joint)
        draws=rng.multivariate_normal(np.zeros(12),joint,200000);a=draws[:,:6]@jy.T+draws[:,6:]@jt.T
        b=draws[:,6:]@other.T
        assert_allclose(np.cov(a.T),pp,rtol=.03,atol=3e-5)
        assert_allclose((a-a.mean(0)).T@(b-b.mean(0))/len(a),jt@p@other.T,rtol=.1,atol=1e-5)
        # Forward at the source: R dp + [I,-[y]x] dT has exactly
        # zero dependence on the source pose. It is NOT extra motion evidence.
        from rgbd_common.lie import skew
        assert_allclose(pose[:3,:3]@jt[:3]+np.c_[np.eye(3),-skew(world)],0,atol=1e-15)
        self.assertGreater(np.linalg.norm(pt),0)
    def test_late_point_cross_terms_vs_joint_pushforward(self):
        rng=np.random.default_rng(8);a=rng.normal(size=(9,9));joint=a@a.T*.001
        pose=exp(np.array([.1,.2,.3,.2,-.3,.1]));world=np.array([.3,-.1,1.])
        pp,pt,j=source_latent_covariance(world,pose,joint[:3,:3],joint[3:,3:],joint[:3,3:])
        push=np.c_[pose[:3,:3].T,j]
        assert_allclose(pp,push@joint@push.T,atol=1e-14);assert_allclose(pt,push@joint[:,3:],atol=1e-14)
    def test_selected_information_inverse_and_gauge_elimination(self):
        a=np.array([[4.,1,1],[1,3,0],[1,0,3.]])
        actual,residuals=selected_covariance(aslinearoperator(a),[0,2])
        assert_allclose(actual,np.linalg.inv(a)[np.ix_([0,2],[0,2])],atol=1e-12)
        self.assertLess(max(residuals),1e-9)
    def test_joint_source_queries_reject_invalid_covariance(self):
        from rgbd_common.source_factors import augmentation_blocks
        for bad in (np.eye(11),np.eye(12)*np.nan,np.diag([-1.]+[1.]*11)):
            with self.assertRaises(ValueError):augmentation_blocks(np.eye(6),np.eye(6),bad)

    def test_source_pose_independence_is_never_silently_assumed(self):
        from rgbd_common.lie import source_map_covariance
        for function in (source_latent_covariance,source_map_covariance):
            with self.assertRaisesRegex(ValueError,'cross covariance'):
                function(np.array([0.,0.,1.]),np.eye(4),np.eye(3)*.001,np.eye(6)*.001)

    def test_exported_source_weights_relinearize_old_fusion_covariance(self):
        import tempfile
        from pathlib import Path
        from parallel_fixtures import Scene,configuration
        from rgbd_common.fusion import fuse_frame_zero
        from rgbd_common.backends.torch_cuda import Backend
        from rgbd_common.source_factors import export_source_factors
        from rgbd_common.lie import State
        scene=Scene(3,2);config=configuration();backend=Backend('cpu');model,_=fuse_frame_zero(scene,config,backend)
        model.append([[.1,.2,.8]],[[.2,.3,.4]],[0],np.eye(6)[None]*1e-9,[128],[[128,192]])
        states=[State(exp(np.array([.05*k,.01*k,0,.03*k,.02*k,0]))[None],np.zeros((1,6))) for k in range(3)]
        with tempfile.TemporaryDirectory() as root:
            export_source_factors(model,states,scene,config,backend,root)
            with np.load(Path(root)/'source_coordinate_gaussian.npz') as data:
                covariance=data['conditional_covariance'];total=covariance@np.linalg.inv(config.map_covariance)
                self.assertGreater(np.linalg.norm(data['mean_offset_from_accepted'][-1]),.1)
                self.assertGreater(np.trace(covariance[-1]),1e-5)
            for path in Path(root).glob('source_??????_???.npz'):
                with np.load(path) as data:np.add.at(total,data['point_index'],data['fusion_weight'])
            assert_allclose(total,np.broadcast_to(np.eye(6),total.shape),atol=1e-12)

    def test_overlapping_raw_pixel_noise_is_correlated(self):
        weights=np.array([[.5,.5,0],[0,.5,.5]]);raw=np.eye(3)*.01
        cov=weights@raw@weights.T
        self.assertGreater(cov[0,1],0);self.assertFalse(np.allclose(cov,np.diag(np.diag(cov))))
