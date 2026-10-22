import unittest
import numpy as np
from numpy.testing import assert_allclose
import torch
from rgbd_common.backends.torch_cuda import Backend,numpy,stable_segment_sum
from rgbd_common.projection_cuda import transport_and_project,build_pixel_bins_and_zbuffer
from rgbd_common.lie import exp
from parallel_fixtures import Scene


@unittest.skipUnless(torch.cuda.is_available(),'Host CUDA access is required; sandbox CPU results are not GPU parity')
class GPUParityTests(unittest.TestCase):
    def test_actual_matching_IEKF_reduction_and_UKF_moments_vs_float64_CPU(self):
        from rgbd_common.fusion import fuse_frame_zero
        from rgbd_common.matching_cuda import match_rgbd
        from rgbd_common.measurement_problem import MeasurementProblem
        from rgbd_common.lie import State
        from iekf.measurement import reduce_information
        from ukf.measurement import measurement_moments
        from parallel_fixtures import configuration
        s=Scene();config=configuration();model,_=fuse_frame_zero(s,config,Backend('cpu','float64'))
        state=State(exp(np.array([.003,-.002,.001,.001,-.001,.002]))[None],np.zeros((1,6)));values=[]
        for device,dtype in (('cpu','float64'),('cuda','float32')):
            backend=Backend(device,dtype);matches,*_=match_rgbd(model,state,s.observation(1,0),config,backend)
            problem=MeasurementProblem(model,[matches],backend);information,score=reduce_information(problem,state)
            a,m,b,report=measurement_moments(problem,state,np.eye(12)*1e-7)
            values.append((numpy(matches.point),numpy(information),numpy(score),a,numpy(m),numpy(b),report['active_factors']))
        for first,second in zip(*values):assert_allclose(first,second,rtol=2e-5,atol=2e-5)
    def test_transport_projection_and_deterministic_depth_buffer(self):
        rng=np.random.default_rng(21);xyz=rng.normal(size=(1000,3))*.1;xyz[:,2]+=1
        xyz[:10,2]=-1;group=np.arange(1000)%2
        poses=np.stack([exp(np.array([.01,.02,.03,.2,-.1,.3])),np.eye(4)])
        camera=Scene().cameras[0];results=[]
        for device,dtype in (('cpu','float64'),('cuda','float32')):
            b=Backend(device,dtype);p=transport_and_project(b.tensor(xyz,b.input_dtype),b.tensor(group,torch.long),b.tensor(poses,b.input_dtype),camera)
            bins=build_pixel_bins_and_zbuffer(p,camera);results.append((numpy(p['uv']),numpy(p['depth']),numpy(p['valid']),numpy(bins['owner'])))
        assert_allclose(results[0][0],results[1][0],rtol=1e-5,atol=2e-5);assert_allclose(results[0][1],results[1][1],atol=1e-6)
        assert_allclose(results[0][2],results[1][2]);assert_allclose(results[0][3],results[1][3])
    def test_discrete_projection_is_bitwise_invariant_to_point_batch_size(self):
        rng=np.random.default_rng(812);n=150001;xyz=rng.normal(size=(n,3));xyz[:,2]+=3
        camera=Scene().cameras[0];camera.world_to_camera=exp(np.array([.1,.2,.3,.2,-.1,.3]));camera.width=640;camera.height=480;camera.intrinsics=np.array([500.,501.,319.5,239.5])
        poses=np.stack([exp(np.array([.02,.01,.03,.2,.1,-.1])),np.eye(4)]);group=np.arange(n)%2
        b=Backend('cuda');point=b.tensor(xyz,b.input_dtype);gid=b.tensor(group,torch.long);t=b.tensor(poses,b.input_dtype)
        full=transport_and_project(point,gid,t,camera)
        subset=np.sort(rng.choice(n,23719,replace=False));index=b.tensor(subset,torch.long)
        part=transport_and_project(point[index],gid[index],t,camera)
        for key in ('world','camera','uv','depth','pixel','valid'):self.assertTrue(torch.equal(full[key][index],part[key]),key)

    def test_sigma_projection_is_bitwise_invariant_to_point_batch_size(self):
        from ukf.measurement import project_sigma
        rng=np.random.default_rng(715);n=2003;backend=Backend('cuda');camera=Scene().cameras[0]
        camera.world_to_camera=exp(np.array([.1,.2,.3,.2,-.1,.3]));xyz=rng.normal(size=(n,3))*.1;xyz[:,2]+=1
        poses=np.array([[exp(rng.normal(size=6)*.02),exp(rng.normal(size=6)*.02)] for _ in range(25)])
        point=backend.tensor(xyz);rgb=backend.tensor(rng.uniform(size=(n,3)));group=backend.tensor(np.arange(n)%2,torch.long);pose=backend.tensor(poses)
        value,valid=project_sigma(pose,point,rgb,group,camera,backend)
        ids=backend.tensor(np.sort(rng.choice(n,237,replace=False)),torch.long)
        partial,pvalid=project_sigma(pose,point[ids],rgb[ids],group[ids],camera,backend)
        self.assertTrue(torch.equal(value[:,ids],partial));self.assertTrue(torch.equal(valid[:,ids],pvalid))

    def test_segmented_float64_reduction_repeatability(self):
        b=Backend('cuda');rng=np.random.default_rng(8);x=b.tensor(rng.normal(size=(1000,12)));keys=b.tensor(np.arange(1000)%3,torch.long)
        a=stable_segment_sum(x,keys,3);other=stable_segment_sum(x,keys,3)
        self.assertTrue(torch.equal(a,other));reference=np.array([numpy(x)[np.arange(1000)%3==g].sum(0) for g in range(3)])
        assert_allclose(numpy(a),reference,atol=1e-12)
