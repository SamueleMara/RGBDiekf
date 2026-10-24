import unittest
import numpy as np
import torch
from pointwise_rgbd.grouped_covariance import CovarianceOperator
from pointwise_rgbd.grouped_point_iekf import basis,reduced_inverse
from pointwise_rgbd.process6 import transition,process_covariance
from rgbd_common.backends.torch_cuda import Backend


class CovarianceOperatorTests(unittest.TestCase):
    def check_operator(self,device):
        rng=np.random.default_rng(218);n=4;backend=Backend(device)
        xyz=backend.tensor(rng.normal(size=(n,3)));member=np.zeros(n,dtype=np.int64)
        g=basis(xyz,backend.tensor(member,torch.long),xyz.mean(0)[None]);G=g.cpu().numpy().reshape(6*n,12)
        factors=rng.normal(size=(n,6,6));p=factors@factors.transpose(0,2,1)+np.eye(6)
        P=np.zeros((6*n,6*n))
        for i in range(n):P[i*6:i*6+6,i*6:i*6+6]=p[i]
        op=CovarianceOperator(p,g,member,backend,1e-10)
        for reverse in (False,True,False):
            f=transition(.04,reverse);q=process_covariance(.04,np.eye(3)*.01,np.eye(3)*.001,reverse)
            F=np.kron(np.eye(n),f);Q=np.kron(np.eye(n),q);P=F@P@F.T+Q
            pp=np.stack([P[6*i:6*i+6,6*i:6*i+6] for i in range(n)])
            op.prediction(f,q);B=op.information(backend.tensor(pp)).cpu().numpy()[0]
            np.testing.assert_allclose(B,G.T@np.linalg.solve(P,G),rtol=1e-8,atol=1e-8)
            h=rng.normal(size=(n,3,6));h[:,:,3:]=0;a=h.transpose(0,2,1)@h
            A=np.zeros_like(P)
            for i in range(n):A[6*i:6*i+6,6*i:6*i+6]=a[i]
            M=np.linalg.inv(B+G.T@A@G);U=np.eye(n*6)-G@M@G.T@A
            P=U@P@U.T+G@M@G.T@A@G@M@G.T
            post,cross=op.update(backend.tensor(pp),backend.tensor(a),backend.tensor(M[None]))
            for i in range(n):np.testing.assert_allclose(post[i].cpu(),P[6*i:6*i+6,6*i:6*i+6],rtol=1e-8,atol=1e-8)
            self.assertGreater(max(r['max_relative_residual'] for r in op.last_solve),0)
        self.assertEqual(len(op.history),3)

    def test_matches_dense_multistep_covariance_and_information(self):self.check_operator('cpu')

    @unittest.skipUnless(torch.cuda.is_available(),'Host CUDA required')
    def test_cuda_matches_dense_multistep_covariance_and_information(self):self.check_operator('cuda')


if __name__=='__main__':unittest.main()
