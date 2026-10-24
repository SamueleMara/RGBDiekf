import unittest
import numpy as np
import torch
from pointwise_rgbd.grouped_point_iekf import basis,joseph_blocks,reduced_inverse,correct
from pointwise_rgbd.config import load_config
from pointwise_rgbd.free_phase1 import ForegroundConstraint
from rgbd_common.backends.torch_cuda import Backend
from tests.grouped_fixtures import GroupScene


class FreeGroupedPoints(unittest.TestCase):
    def test_joseph_marginals_and_cross_blocks_match_dense(self):
        torch.set_num_threads(1);torch.manual_seed(23);n=5
        xyz=torch.randn(n,3,dtype=torch.float64);member=torch.zeros(n,dtype=torch.long)
        g=basis(xyz,member,xyz.mean(0)[None]);gg=g.reshape(6*n,12)
        z=torch.randn(n,6,6,dtype=torch.float64);p=z@z.transpose(1,2)+torch.eye(6)
        h=torch.randn(n,3,6,dtype=torch.float64);h[:,:,3:]=0;a=h.transpose(1,2)@h
        info=(g.transpose(1,2)@(torch.linalg.inv(p)+a)@g).sum(0)[None]
        m,_=reduced_inverse(info,(g.transpose(1,2)@g).sum(0)[None])
        post,coupling=joseph_blocks(p,a,g,member,m,1)
        P=torch.block_diag(*p);H=torch.block_diag(*h);gain=gg@m[0]@gg.T@H.T
        expected=(torch.eye(6*n)-gain@H)@P@(torch.eye(6*n)-gain@H).T+gain@gain.T
        actual=P.clone();v=coupling['V'].reshape(6*n,12);M=m[0];C=coupling['C'][0]
        actual+=gg@C@gg.T-gg@M@v.T-v@M@gg.T
        self.assertTrue(torch.allclose(actual,expected,atol=1e-10,rtol=1e-10))
        for i in range(n):self.assertTrue(torch.allclose(post[i],expected[6*i:6*i+6,6*i:6*i+6],atol=1e-10))
        self.assertGreater(float(expected[:6,6:12].abs().max()),0)

    def test_singleton_structural_nullspace(self):
        x=torch.tensor([[0.,0.,1.]],dtype=torch.float64);g=basis(x,torch.zeros(1,dtype=torch.long),x)
        gram=g.transpose(1,2)@g;m,r=reduced_inverse(gram,gram)
        self.assertEqual(int(r[0]),6)
        self.assertTrue(torch.allclose(g@m@g.transpose(1,2),torch.eye(6,dtype=torch.float64)[None]))

    def test_empty_update_preserves_individual_states(self):
        cfg=load_config();x=torch.tensor([[0.,0.,1.],[1.,0.,1.],[0.,1.,1.]],dtype=torch.float64)
        member=torch.zeros(3,dtype=torch.long);g=basis(x,member,x.mean(0)[None])
        state=torch.cat([x,torch.randn(3,3,dtype=torch.float64)],1);p=torch.eye(6,dtype=torch.float64).repeat(3,1,1)
        result,cov,_,report=correct(state,p,[],g,member,cfg)
        self.assertTrue(torch.equal(result,state));self.assertTrue(torch.equal(cov,p))
        self.assertEqual(report['prior_count'],1)

    def test_background_constraint_and_occlusion(self):
        scene=GroupScene();f=scene.observation(0,0);cfg=load_config();backend=Backend('cpu')
        f.object_only_no_external_occluders=True
        f.depth[0,0]=1.
        state=backend.tensor([[-.475,-.355,1.,0,0,0],[-.95,-.71,2.,0,0,0]])
        member=backend.tensor([0,1],torch.long);p=backend.tensor(np.tile(np.eye(6)*1e-6,(2,1,1)))
        check=ForegroundConstraint([f],p,member,2,cfg,backend)
        self.assertFalse(bool(check(state)[0]));self.assertTrue(bool(check(state)[1]))
        f.depth[0,0]=.01
        self.assertFalse(bool(check(state)[0])) # background depth cannot rescue it
        f.object_only_no_external_occluders=None
        self.assertTrue(bool(check(state).all()))

    @unittest.skipUnless(torch.cuda.is_available(),'Host CUDA required')
    def test_cuda_correction_matches_cpu(self):
        from pointwise_rgbd.observations import samples
        scene=GroupScene();cfg=load_config();raw=samples(scene.observation(0,0),cfg)
        s=np.c_[raw.xyz,np.zeros_like(raw.xyz)];s[:,0]+=.002
        p=np.tile(np.eye(6)*.001,(len(s),1,1));outputs=[]
        for device in ('cpu','cuda'):
            backend=Backend(device);member=backend.tensor(np.arange(len(s))//32,torch.long)
            centers=np.stack([raw.xyz[np.arange(len(s))//32==g].mean(0) for g in range(int(member.max())+1)])
            g=basis(backend.tensor(raw.xyz),member,backend.tensor(centers))
            match={'point':backend.tensor(np.arange(len(s)),torch.long),'pixel':raw.pixel,'camera_index':0,'camera':scene.cameras[0],'z':backend.tensor(raw.z),'R':backend.tensor(raw.R)}
            state,cov,_,_=correct(backend.tensor(s),backend.tensor(p),[match],g,member,cfg)
            outputs.append((state.cpu().numpy(),cov.cpu().numpy()))
        for a,b in zip(*outputs):self.assertTrue(np.allclose(a,b,atol=1e-9,rtol=1e-8))


if __name__=='__main__':unittest.main()
