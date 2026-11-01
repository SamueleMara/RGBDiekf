"""One-point motion across an interpolation-cell boundary: actual estimator,
not merely a forward warp evaluation. All numbers specify this toy sensor only.
"""
import unittest
import numpy as np
import torch
from tests.test_temporal_warp import rig,frame
from estimator.temporal_reference import ReferenceTable
from estimator.temporal_zbuffer import first_hit_pixel_owner
from estimator.temporal_rgbd_terms import build_temporal_rows,raw_values,sensor_covariance
from estimator.temporal_simplified_iekf import update_points_temporal_simplified
from estimator.temporal_full_graph import TemporalWindowGraph
from estimator.temporal_run import add_full_rows,refresh_full_reference_support
from estimator.projective_sensor import SensorNoise
from estimator.point_kalman_cuda import predict,transition,process_noise

MOTION_WITNESSES=[]


class TemporalMotionTests(unittest.TestCase):
    def test_positive_and_negative_motion_refresh_reference_cells(self):
        for dx in (.1,-.1):
            for device in ['cpu']+(['cuda'] if torch.cuda.is_available() else []):
                with self.subTest(dx=dx,device=device):self.check_motion(dx,device)

    def check_motion(self,dx,device):
        r=rig(device);f0=frame(device);f1=frame(device);f1.k=1
        f0.mask[1]=False;f1.mask[1]=False;f0.depth[:]=2.;f1.depth[:]=2.
        # camera fx=4, z=2 -> image shift = 2*dx pixels
        f1.rgb[0,:,:,0]-=.02*2*dx;f1.rgb[0,:,:,1]-=.03*2*dx
        s=torch.tensor([[0.,0.,2.,0.,0.,0.]],dtype=torch.float64,device=device)
        P=torch.diag(torch.tensor([1/48,1/48,1e-8,1.,1.,1.],dtype=torch.float64,device=device))[None]
        noise=SensorNoise(torch.ones(3,dtype=torch.float64,device=device)*1e-8,1e-8,0.)
        warp=torch.eye(4,dtype=torch.float64,device=device)*1e-8;qc=torch.eye(3,dtype=torch.float64,device=device)*.01
        refs=ReferenceTable(2,1,device,torch.float64);refs.initialize_from_source_catalogue(np.array([[0,0,27]]),0,s)
        policy={'same_surface_n_sigma':3.,'same_surface_max_incidence_deg':80.}
        mean,prior=predict(s,P,.04,qc)
        def build(x):
            own=first_hit_pixel_owner(x[:,:3],r,f1)
            rows,reason,life=build_temporal_rows(x,own,f1,refs,{0:f0},r,noise,policy,warp,1,3)
            return rows,own,reason,life
        simple=update_points_temporal_simplified(mean,prior,build,f1,max_iters=10)
        self.assertLess(abs(float(simple[0][0,0])-dx),.002)
        self.assertTrue(any(h['accepted'] for h in simple[-1]['history']))
        g=TemporalWindowGraph(device);val=raw_values(f0,torch.tensor(0),torch.tensor(27))
        g.add_coupled_anchor_factors(0,0,(0,0,27),val,sensor_covariance(val,noise),g.tensor([3.,3.]),r,0,np.eye(2)/12,np.eye(3))
        g.add_process_factors(0,0,1,transition(.04,device=device),process_noise(.04,qc),mean[0])
        rows,*_=build(mean);add_full_rows(g,rows,1,f1,{0:f0},r,noise,warp)
        sensor_count_before=g.pixels.total_sensor_factors
        result=g.solve_sparse_map(max_iters=10,accept=lambda graph:refresh_full_reference_support(graph,{0:f0,1:f1},r,noise,policy))
        state,j=g.state(0,1);anchor,_=g.state(0,0)
        relative=float(state[0]-anchor[0]);velocity=float(state[3])
        self.assertLess(abs(relative-dx),.002)
        self.assertLess(abs(velocity-dx/.04),.08)
        self.assertTrue(all(n==1 for n in g.pixels.sensor_counts.values()))
        if dx>0:self.assertGreater(g.pixels.total_sensor_factors,sensor_count_before)
        MOTION_WITNESSES.append({'device':device,'true_dx_m':dx,'simplified_dx_m':float(simple[0][0,0]),'full_relative_dx_m':relative,
            'full_velocity_m_s':velocity,'true_velocity_m_s':dx/.04,'newly_touched_sensor_pixels':g.pixels.total_sensor_factors-sensor_count_before,
            'full_covariance_trace':float(g.covariance(j,j).trace()),'converged':result['converged']})
