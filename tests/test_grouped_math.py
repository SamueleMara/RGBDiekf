import tempfile,unittest
from pathlib import Path
import numpy as np
import torch
from numpy.testing import assert_allclose,assert_array_equal
from rgbd_common.lie import exp as np_exp,log,inverse
from rgbd_common.backends.torch_cuda import Backend
from rgbd_common.projection_cuda import Camera
from pointwise_rgbd.group_pose import exp,transport,point_jacobian,left_jacobian,chart_reset,cross_position_covariance
from pointwise_rgbd.group_process import matrices,predict,initialize,load_group_noise
from pointwise_rgbd.group_iekf import correct
from pointwise_rgbd.projection import project
from pointwise_rgbd.config import load_config


class GroupMathTests(unittest.TestCase):
    def setUp(self):self.b=Backend('cpu');self.noise=load_group_noise();self.config=load_config()
    def test_group_jacobian_and_left_chart_reset_finite_difference(self):
        point=np.array([.2,-.3,1.7]);eps=1e-7
        jac=point_jacobian(self.b.tensor(point)).numpy()
        for j in range(6):
            d=np.eye(6)[j]*eps
            numeric=((np_exp(d)@np.r_[point,1])[:3]-(np_exp(-d)@np.r_[point,1])[:3])/(2*eps)
            assert_allclose(jac[:,j],numeric,atol=2e-9)
        d=np.array([.3,-.2,.1,.2,-.3,.1]);base=np_exp(d);reset=left_jacobian(self.b.tensor(d)).numpy()
        for j in range(6):
            step=np.eye(6)[j]*eps
            numeric=(log(np_exp(d+step)@inverse(base))-log(np_exp(d-step)@inverse(base)))/(2*eps)
            assert_allclose(reset[:,j],numeric,atol=2e-9)
    def test_signed_process_and_reoriented_gaussian_factor(self):
        xi=self.b.tensor([[.3,-.2,.1,.2,.5,-.4]])
        f,q=matrices(xi,.17,self.noise.acceleration_spectral_density);back,qback=matrices(xi,-.17,self.noise.acceleration_spectral_density)
        assert_allclose((back@f).numpy(),np.eye(12)[None],atol=1e-13)
        assert_allclose(qback.numpy(),(back@q@back.transpose(-1,-2)).numpy(),atol=1e-16)
        self.assertGreater(np.linalg.eigvalsh(q.numpy()).min(),0)
        self.assertLess(float(qback[0,0,6]),0);self.assertGreater(float(q[0,0,6]),0)
        base=np_exp([.1,.2,.3,.2,-.1,.3]);eps=1e-7
        for h in (.17,-.17):
            f,_=matrices(xi,h,self.noise.acceleration_spectral_density);nominal=np_exp(h*xi.numpy()[0])@base
            for j in range(12):
                e=np.eye(12)[j]*eps;values=[]
                for sign in (1,-1):
                    pose=np_exp(h*(xi.numpy()[0]+sign*e[6:]))@np_exp(sign*e[:6])@base
                    values.append(np.r_[log(pose@inverse(nominal)),sign*e[6:]])
                assert_allclose(f[0,:,j].numpy(),(values[0]-values[1])/(2*eps),atol=3e-9)
            error=np.linspace(-.1,.1,12);reversed_error=-back.numpy()[0]@error
            self.assertAlmostEqual(float(error@np.linalg.solve(q.numpy()[0],error)),float(reversed_error@np.linalg.solve(qback.numpy()[0],reversed_error)),places=6)
    def test_exact_rigidity_prediction_update_directions_and_serialization(self):
        canonical=self.b.tensor([[.1,.1,1.],[-.2,.2,1.],[.2,-.3,1.2],[0.,0.,1.4]])
        membership=self.b.tensor(np.zeros(4,np.int64),torch.long);dist=torch.cdist(canonical,canonical).numpy()
        for h in (.04,-.04):
            poses,xi,p=initialize(1,self.noise,self.b);xi[:]=self.b.tensor([[.1,-.2,.3,.5,-.2,.7]])
            for _ in range(8):
                poses,xi,p=predict(poses,xi,p,h,self.noise);xyz=transport(canonical,membership,poses)
                assert_allclose(torch.cdist(xyz,xyz).numpy(),dist,atol=1e-14)
                previous=(poses.clone(),xi.clone(),p.clone())
                poses,xi,p,report=correct(poses,xi,p,canonical,membership,[],self.config,self.b)
                for actual,expected in zip((poses,xi,p),previous):assert_array_equal(actual.numpy(),expected.numpy())
            with tempfile.TemporaryDirectory() as root:
                path=Path(root)/'checkpoint.npz';np.savez(path,poses=poses.numpy(),twists=xi.numpy(),covariance=p.numpy())
                with np.load(path) as f:restored=[self.b.tensor(f[name]) for name in ('poses','twists','covariance')]
                a=predict(poses,xi,p,h,self.noise);b=predict(*restored,h,self.noise)
                for x,y in zip(a,b):assert_array_equal(x.numpy(),y.numpy())
    def test_observability_one_two_three_points_and_induced_cross_covariance(self):
        for n,expected in [(1,3),(2,5),(3,6)]:
            points=self.b.tensor([[0.,0.,1.],[.3,0.,1.],[0.,.4,1.]])[:n]
            h=point_jacobian(points).reshape(-1,6);singular=torch.linalg.svdvals(h).numpy()
            rank=np.linalg.matrix_rank(h.numpy());self.assertEqual(rank,expected)
        x=self.b.tensor([.1,.2,1.]);y=self.b.tensor([-.2,.4,1.]);p=torch.eye(6,dtype=torch.float64)
        cross=cross_position_covariance(x,y,p)
        self.assertGreater(float(torch.linalg.vector_norm(cross)),1.)
    def test_fixed_prior_manifold_correction_and_pose_rank(self):
        camera=Camera('test',640,480,np.array([500.,500.,320.,240.]),np.eye(4),0,np.inf)
        points=self.b.tensor([[-.2,-.1,1.],[-.2,.1,1.],[.2,-.1,1.],[.2,.1,1.]])
        membership=self.b.tensor(np.zeros(4,np.int64),torch.long)
        poses,xi,p=initialize(1,self.noise,self.b);poses,xi,p=predict(poses,xi,p,.04,self.noise)
        truth=exp(self.b.tensor([[.003,-.002,.005,.01,-.005,.004]]));xyz=transport(points,membership,truth);z,_=project(xyz,camera)
        match={'pixel':np.arange(4),'point':self.b.tensor(np.arange(4),torch.long),'z':z,'R':self.b.tensor(self.config.measurement_covariance(z[:,2].numpy())),'camera':camera}
        result=correct(poses,xi,p,points,membership,[match],self.config,self.b)
        after,velocity,cov,report=result
        self.assertLess(report['energy_after'][0],report['energy_before'][0]);self.assertEqual(report['pose_rank'][0],6)
        self.assertTrue(np.all(np.diff(report['energy_history'],axis=0)<=0))
        assert_allclose(torch.cdist(transport(points,membership,after),transport(points,membership,after)).numpy(),torch.cdist(points,points).numpy(),atol=1e-14)
        self.assertGreater(np.linalg.eigvalsh(cov.numpy()).min(),0)
        self.assertGreater(float(torch.linalg.vector_norm(cov[0,:6,6:])),0)
        from scipy.linalg import block_diag
        from rgbd_common.lie import skew,left_jacobian as numpy_jacobian
        d=log(after[0].numpy()@inverse(poses[0].numpy()));c=block_diag(numpy_jacobian(d),np.eye(6));ci=np.linalg.inv(c)
        information=ci.T@np.linalg.inv(p[0].numpy())@ci
        world=transport(points,membership,after).numpy()
        for point,measurement_cov in zip(world,match['R'].numpy()):
            _,jx=project(self.b.tensor(point[None]),camera);hg=jx.numpy()[0,:,:3]@np.c_[np.eye(3),-skew(point)]
            h=np.c_[hg,np.zeros((3,6))];information+=h.T@np.linalg.solve(measurement_cov,h)
        inverse_residual=information@cov[0].numpy()-np.eye(12)
        backward_error=float(np.linalg.norm(inverse_residual)/(np.linalg.norm(information)*np.linalg.norm(cov[0].numpy())))
        self.assertLess(backward_error,12*np.finfo(float).eps)
        self.metrics={'posterior_reference_inverse_residual_max':float(abs(inverse_residual).max()),'normalized_backward_error':backward_error,'roundoff_budget':12*np.finfo(float).eps}

    def test_duplicate_raw_factors_rejected_and_projection_group_jacobian(self):
        camera=Camera('test',640,480,np.array([500.,500.,320.,240.]),np.eye(4),0,np.inf)
        points=self.b.tensor([[.2,.1,1.]])
        h,j=project(points,camera);analytic=(j[:,:,:3]@point_jacobian(points)).numpy()[0]
        eps=1e-7
        for column in range(6):
            e=np.eye(6)[column]*eps
            a=(np_exp(e)@np.r_[points.numpy()[0],1])[:3]
            b=(np_exp(-e)@np.r_[points.numpy()[0],1])[:3]
            numeric=(project(self.b.tensor(a[None]),camera)[0]-project(self.b.tensor(b[None]),camera)[0]).numpy()[0]/(2*eps)
            assert_allclose(analytic[:,column],numeric,atol=1e-6)
        poses,xi,p=initialize(1,self.noise,self.b);poses,xi,p=predict(poses,xi,p,.04,self.noise)
        match={'pixel':np.array([90]),'point':self.b.tensor([0],torch.long),'z':h,'R':self.b.tensor(self.config.measurement_covariance(h[:,2].numpy())),'camera':camera}
        with self.assertRaisesRegex(ValueError,'Duplicate raw'):
            correct(poses,xi,p,points,self.b.tensor([0],torch.long),[match,match],self.config,self.b)

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA hardware required')
    def test_cpu_cuda_process_and_group_correction_agree(self):
        results=[]
        for device in ('cpu','cuda'):
            b=Backend(device);poses,xi,p=initialize(2,self.noise,b);xi[:]=b.tensor([[.1,.2,.3,.4,.3,.2],[.3,.1,.2,.2,.1,.4]])
            predicted=predict(poses,xi,p,-.04,self.noise)
            camera=Camera('test',640,480,np.array([500.,500.,320.,240.]),np.eye(4),0,np.inf)
            points=b.tensor([[-.2,-.1,1.],[-.2,.1,1.],[.2,-.1,1.],[.2,.1,1.]])
            membership=b.tensor(np.array([0,0,1,1]),torch.long)
            target,_=project(transport(points,membership,predicted[0]),camera);target=target+b.tensor([.2,-.1,.001])
            match={'pixel':np.arange(4),'point':b.tensor(np.arange(4),torch.long),'z':target,'R':b.tensor(self.config.measurement_covariance(target[:,2].cpu().numpy())),'camera':camera}
            result=correct(*predicted,points,membership,[match],self.config,b)
            results.append([x.cpu().numpy() for x in result[:3]])
        # Covariance inversion in an almost unobserved axial-spin direction
        # amplifies FP64 roundoff. Check the conditioning-derived forward-error
        # budget, not an unexplained fixed precision requirement.
        condition=max(float(np.linalg.cond(p).max()) for p in (results[0][2],results[1][2]))
        errors=[];bounds=[]
        for a,b in zip(*results):
            error=float(np.linalg.norm(a-b));bound=np.finfo(float).eps*12*condition*max(float(np.linalg.norm(a)),float(np.linalg.norm(b)))
            self.assertLessEqual(error,bound);errors.append(error);bounds.append(bound)
        self.metrics={'cpu_cuda_output_norm_errors':errors,'conditioning_derived_roundoff_budgets':bounds,'posterior_condition_max':condition,'covariance_max_absolute_difference':float(abs(results[0][2]-results[1][2]).max())}


if __name__=='__main__':unittest.main()
