import unittest
import numpy as np
import torch
from estimator.temporal_selected_inverse_gpu import selected_inverse_blocks_gpu
from estimator.temporal_sparse_gpu import sparse_coo


class SelectedInverseTests(unittest.TestCase):
    def test_indefinite_selected_blocks_match_independent_joint_inverse(self):
        if not torch.cuda.is_available():raise RuntimeError('Actual CUDA required')
        rng=np.random.default_rng(62);A=rng.normal(size=(4,5));B=rng.normal(size=(5,5));H=B@B.T+np.eye(5)
        K=np.block([[np.eye(4),A],[A.T,-H]])
        query=np.zeros((6,9));query[np.arange(6),4+np.array([0,1,2,2,3,4])]=1.
        expected=-query@np.linalg.inv(K)@query.T
        actual,profile=selected_inverse_blocks_gpu(torch.tensor(K,device='cuda').to_sparse_csr(),torch.tensor(query,device='cuda').to_sparse(),3)
        np.testing.assert_allclose(actual.cpu(),np.stack((expected[:3,:3],expected[3:,3:])),atol=5e-12,rtol=5e-12)
        self.assertFalse(profile['dense_global_matrices'])

    def test_shared_temporal_pixels_and_source_coupling_match_joint_sparse_solves(self):
        if not torch.cuda.is_available():raise RuntimeError('Actual CUDA required')
        from tests.test_temporal_sparse_gpu import fixture_graph
        from estimator.temporal_sparse_solve import SparseJointFactorization
        graph=fixture_graph('cuda',True)
        with SparseJointFactorization(graph.assemble_sparse_linearized_problem_gpu()) as factor:
            q=graph.state_query_gpu(1);reduced=torch.sparse.mm(q,factor.T).coalesce();ix=reduced.indices().clone();ix[1]+=factor.A.shape[0]
            query=sparse_coo(ix,reduced.values(),(q.shape[0],factor.K.shape[0]))
            actual,_=selected_inverse_blocks_gpu(factor.K,query,12)
            expected=factor.extract_joint_selected_marginals_gpu(q,12)
            torch.testing.assert_close(actual,expected,atol=5e-12,rtol=5e-12)


if __name__=='__main__':unittest.main()
