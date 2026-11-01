import unittest
from types import SimpleNamespace
import numpy as np
import torch
from estimator.projective_camera import CameraRig
from estimator.temporal_zbuffer import first_hit_pixel_owner, numpy_first_hit
from estimator.temporal_rgbd_terms import temporal_prediction, validate_reference_patch
from estimator.temporal_reference import ReferenceTable
from estimator.temporal_simplified_iekf import update_points_temporal_simplified
from estimator.projective_sensor import SensorNoise


def rig(device='cpu'):
    cams=[SimpleNamespace(name=f'c{c}',width=8,height=8,intrinsics=np.array([4.,4.,3.,3.]),near=.01,world_to_camera=np.eye(4)) for c in range(2)]
    cams[1].world_to_camera[0,3]=.1
    return CameraRig(cams,device)


def frame(device='cpu'):
    yy,xx=torch.meshgrid(torch.arange(8),torch.arange(8),indexing='ij')
    f=SimpleNamespace(H=8,W=8,k=0)
    f.rgb=torch.stack((xx*.02+yy*.01,xx*.03-yy*.01,yy*.04),-1).double().repeat(2,1,1,1).to(device)
    f.depth=(2.+xx*.001+yy*.002).double().repeat(2,1,1).to(device)
    f.mask=torch.ones((2,8,8),dtype=torch.bool,device=device);f.raw_valid=f.mask.clone()
    return f


class TemporalWarpTests(unittest.TestCase):
    def test_sparse_and_dense_owner_queries_are_identical(self):
        from estimator.temporal_zbuffer import selected_owners,owner_at,compare_owner_maps
        for device in ['cpu']+(['cuda'] if torch.cuda.is_available() else []):
            gen=torch.Generator().manual_seed(73);points=torch.randn((83,3),generator=gen,dtype=torch.float64);points[:,2]+=3;points=points.to(device)
            a=first_hit_pixel_owner(points,rig(device),frame(device));b=first_hit_pixel_owner(points,rig(device),frame(device),dense=False)
            for x,y in zip(selected_owners(a),selected_owners(b)):torch.testing.assert_close(x,y,atol=0,rtol=0)
            torch.testing.assert_close(a.reason,b.reason,atol=0,rtol=0)
            self.assertEqual(compare_owner_maps(a,b)['switched_pixels'],0)
            for c in range(2):
                for p in range(64):self.assertEqual(owner_at(b,c,p),int(a.owner_id[c,p//8,p%8]))
    def test_first_hit_tie_depth_swap_numpy(self):
        r=rig();f=frame();p=torch.tensor([[0.,0.,2.],[0.,0.,2.],[0.,0.,3.]],dtype=torch.float64)
        ids=torch.tensor([7,2,9]);o=first_hit_pixel_owner(p,r,f,ids)
        self.assertEqual(int(o.owner_id[0,3,3]),2)
        self.assertEqual(int(o.counts[0,3,3]),3);self.assertEqual(int(o.ties[0,3,3]),2)
        np.testing.assert_array_equal(o.owner_id.numpy(),numpy_first_hit(o.uv.numpy(),o.z.numpy(),ids.numpy(),8,8,r.near.numpy()))
        p[2,2]=1.;oo=first_hit_pixel_owner(p,r,f,ids)
        self.assertEqual(int(oo.owner_id[0,3,3]),9)
        self.assertEqual(int(oo.reason[0,0]),6)

    def test_random_numpy_parity(self):
        for seed in range(4):
            g=torch.Generator().manual_seed(seed);p=torch.randn((150,3),generator=g,dtype=torch.float64);p[:,2]+=3
            o=first_hit_pixel_owner(p,rig(),frame())
            np.testing.assert_array_equal(o.owner_id,numpy_first_hit(o.uv.numpy(),o.z.numpy(),np.arange(150),8,8,rig().near.numpy()))

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA unavailable')
    def test_cuda_owner_parity(self):
        p=torch.randn((100,3),dtype=torch.float64);p[:,2]+=3
        a=first_hit_pixel_owner(p,rig(),frame());b=first_hit_pixel_owner(p.cuda(),rig('cuda'),frame('cuda'))
        torch.testing.assert_close(a.owner_id,b.owner_id.cpu(),rtol=0,atol=0)

    def test_two_time_and_latent_jacobian(self):
        r=rig();c=torch.tensor([0]);a=torch.tensor([[.11,.14,2.1,.2,0.,0.]],dtype=torch.float64)
        b=torch.tensor([[.03,.08,2.,0.,0.,0.]],dtype=torch.float64)
        p=torch.tensor([[3.,3.]],dtype=torch.float64)
        values=torch.tensor([[[.1,.2,.3,2.],[.2,.3,.4,2.01],[.15,.25,.35,2.02],[.3,.4,.5,2.03]]],dtype=torch.float64)
        h,ht,hr,q,w,*_=temporal_prediction(r,a,b,c,p,values)
        eps=1e-6
        for arr,H in ((a,ht),(b,hr)):
            for col in range(6):
                plus=arr.clone();minus=arr.clone();plus[0,col]+=eps;minus[0,col]-=eps
                args1=(plus,b) if arr is a else (a,plus);args2=(minus,b) if arr is a else (a,minus)
                fd=(temporal_prediction(r,*args1,c,p,values)[0]-temporal_prediction(r,*args2,c,p,values)[0])/(2*eps)
                torch.testing.assert_close(fd,H[:,:,col],atol=3e-9,rtol=3e-7)
        for j in range(4):
            for ch in range(4):
                v1=values.clone();v2=values.clone();v1[0,j,ch]+=eps;v2[0,j,ch]-=eps
                fd=(temporal_prediction(r,a,b,c,p,v1)[0]-temporal_prediction(r,a,b,c,p,v2)[0])/(2*eps)
                want=torch.zeros_like(h);want[0,ch]=w[0,j]
                torch.testing.assert_close(fd,want,atol=1e-9,rtol=1e-7)
        aa=a.clone();aa[0,0]+=1e-12
        self.assertLess(float((temporal_prediction(r,aa,b,c,p,values)[0]-h).abs().max()),1e-11)

    def test_affine_translations_and_negative_time(self):
        r=rig();c=torch.tensor([0]);ref=torch.tensor([[0.,0.,2.,0.,0.,0.]],dtype=torch.float64)
        for shift in (.2,1.,2.3,-.2):
            s=ref.clone();s[0,0]=shift*.5
            pixel=torch.tensor([[5.,4.]],dtype=torch.float64)
            q=pixel-torch.tensor([[shift,0.]],dtype=torch.float64)
            base=q.floor();offset=torch.tensor([[0,0],[1,0],[0,1],[1,1]])
            xy=base[:,None]+offset
            values=torch.stack((xy[...,0]*.03,xy[...,1]*.02,xy.sum(-1)*.01,2.+xy[...,0]*.001),-1)
            h,ht,*_=temporal_prediction(r,s,ref,c,pixel,values)
            expected=torch.stack((q[:,0]*.03,q[:,1]*.02,q.sum(-1)*.01,2.+q[:,0]*.001),-1)
            torch.testing.assert_close(h,expected,atol=1e-14,rtol=1e-14)
            values[:,:,:3]=.5
            self.assertEqual(float(temporal_prediction(r,s,ref,c,pixel,values)[1][:,:3].abs().max()),0.)

    def test_support_all_four_even_zero_weight(self):
        f=frame();r=rig();q=torch.tensor([[3.,3.]],dtype=torch.float64);c=torch.tensor([0])
        noise=SensorNoise(torch.ones(3,dtype=torch.float64)*.01,.001,0.)
        pol={'same_surface_n_sigma':3.,'same_surface_max_incidence_deg':80.}
        self.assertEqual(int(validate_reference_patch(f,c,q,r,noise,pol)[2][0]),0)
        f.mask[0,4,4]=False
        pix,v,reason,_=validate_reference_patch(f,c,q,r,noise,pol)
        self.assertEqual(int(reason[0]),3);self.assertTrue(bool(torch.isfinite(v).all()));self.assertEqual(float(v.abs().sum()),0.)
        f.mask[0,4,4]=True;f.depth[0,4,4]=float('nan')
        self.assertEqual(int(validate_reference_patch(f,c,q,r,noise,pol)[2][0]),4)
        f.depth[0,4,4]=100
        self.assertEqual(int(validate_reference_patch(f,c,q,r,noise,pol)[2][0]),5)
        q[0,0]=8
        self.assertEqual(int(validate_reference_patch(f,c,q,r,noise,pol)[2][0]),2)

    def test_source_identity_and_handover_bootstrap(self):
        s=torch.tensor([[0.,0.,2.,0.,0.,0.]],dtype=torch.float64)
        tab=ReferenceTable(2,1,'cpu',torch.float64)
        tab.initialize_from_source_catalogue(np.array([[0,0,27]]),0,s)
        self.assertEqual(int(tab.frame[1,0]),-1)
        f=frame();f.k=1;o=first_hit_pixel_owner(s[:,:3],rig(),f)
        tab.commit_accepted_references(o,f,s,1)
        self.assertEqual(int(tab.frame[1,0]),1)
        self.assertEqual(int(tab.source_identity[0,2]),27)

    def test_no_rows_exact_prediction(self):
        from estimator.temporal_rgbd_terms import build_temporal_rows
        from estimator.point_kalman_cuda import predict
        f=frame();f.k=1;r=rig();s=torch.tensor([[0.,0.,2.,.5,0.,0.]],dtype=torch.float64)
        P=torch.eye(6,dtype=torch.float64)[None];m,pp=predict(s,P,-.1,torch.eye(3,dtype=torch.float64)*.01)
        tab=ReferenceTable(2,1,'cpu',torch.float64)
        noise=SensorNoise(torch.ones(3,dtype=torch.float64)*.01,.001,0.)
        def build(x):
            o=first_hit_pixel_owner(x[:,:3],r,f)
            rows,re,li=build_temporal_rows(x,o,f,tab,{},r,noise,{'same_surface_n_sigma':3.,'same_surface_max_incidence_deg':80.},torch.eye(4,dtype=torch.float64)*.001,1,3)
            return rows,o,re,li
        out=update_points_temporal_simplified(m,pp,build,f)
        torch.testing.assert_close(out[0],m,rtol=0,atol=0);torch.testing.assert_close(out[1],pp,rtol=0,atol=0)
        self.assertAlmostEqual(float(m[0,0]),-.05)

    def test_camera_handover_stale_reference_and_occlusion(self):
        from estimator.temporal_rgbd_terms import build_temporal_rows
        r=rig();s=torch.tensor([[0.,0.,2.,0.,0.,0.]],dtype=torch.float64)
        f0,f1,f2=frame(),frame(),frame()
        f0.k=0;f1.k=1;f2.k=2
        f0.mask[1]=False;f2.mask[0]=False
        f0.rgb[0]=.1;f1.rgb[0]=.1;f1.rgb[1]=.8;f2.rgb[1]=.8
        refs=ReferenceTable(2,1,'cpu',torch.float64)
        refs.initialize_from_source_catalogue(np.array([[0,0,27]]),0,s)
        noise=SensorNoise(torch.ones(3,dtype=torch.float64)*.01,.001,0.)
        policy={'same_surface_n_sigma':3.,'same_surface_max_incidence_deg':80.}
        warp=torch.eye(4,dtype=torch.float64)*.01
        own=first_hit_pixel_owner(s[:,:3],r,f1)
        rows,_,life=build_temporal_rows(s,own,f1,refs,{0:f0},r,noise,policy,warp,1,3)
        self.assertEqual(rows.camera.tolist(),[0]);self.assertEqual(life['bootstrap'],1)
        refs.commit_accepted_references(own,f1,s,1)
        own=first_hit_pixel_owner(s[:,:3],r,f2)
        rows,_,_=build_temporal_rows(s,own,f2,refs,{1:f1},r,noise,policy,warp,2,3)
        self.assertEqual(rows.camera.tolist(),[1]);self.assertEqual(rows.reference_frame.tolist(),[1])
        torch.testing.assert_close(rows.y[:,:3],rows.h[:,:3],atol=0,rtol=0)
        # After a gap outside lag, never reuse an old deterministic state.
        f2.k=5
        rows,_,life=build_temporal_rows(s,own,f2,refs,{},r,noise,policy,warp,5,3)
        self.assertEqual(len(rows),0);self.assertEqual(life['reference_outside_lag'],1)

    def test_conservative_domain_rejects_row_deletion(self):
        from estimator.temporal_simplified_iekf import same_domain,row_signature
        from estimator.temporal_rgbd_terms import build_temporal_rows
        r=rig();f=frame();f.k=1;s=torch.tensor([[0.,0.,2.,0.,0.,0.]],dtype=torch.float64)
        refs=ReferenceTable(2,1,'cpu',torch.float64);refs.initialize_from_source_catalogue(np.array([[0,0,27]]),0,s)
        noise=SensorNoise(torch.ones(3,dtype=torch.float64)*.01,.001,0.)
        args=(r,noise,{'same_surface_n_sigma':3.,'same_surface_max_incidence_deg':80.},torch.eye(4,dtype=torch.float64)*.01,1,3)
        own=first_hit_pixel_owner(s[:,:3],r,f)
        a,_,_=build_temporal_rows(s,own,f,refs,{0:frame()},*args)
        f.mask[:]=False
        b,_,_=build_temporal_rows(s,own,f,refs,{0:frame()},*args)
        self.assertFalse(same_domain(row_signature(a),row_signature(b)))
