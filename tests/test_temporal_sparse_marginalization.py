import unittest
import numpy as np
import torch
from scipy.linalg import null_space, qr, solve_triangular
from estimator.temporal_sparse_gpu import SparseLinearizedProblem, sparse_coo
from estimator.temporal_sparse_solve import SparseJointFactorization
from estimator.temporal_sparse_marginalization import marginalize_lag_boundary_sparse_gpu
from estimator.temporal_cudss import SparseNumericalFailure
from tests.test_temporal_sparse_gpu import fixture_graph, dense_oracle_for

MARGINAL_WITNESSES=[]


def independent_qr(A,r,C,c):
    A,r,C,c=[np.asarray(v,dtype=float) for v in (A,r,C,c)]
    if len(C):
        d0=np.linalg.lstsq(C,-c,rcond=None)[0];N=null_space(C)
    else:d0=np.zeros(A.shape[1]);N=np.eye(A.shape[1])
    Q,R=qr(A@N,mode='economic');u=solve_triangular(R,-Q.T@(r+A@d0))
    inverse=solve_triangular(R,np.eye(len(R)));V=N@inverse
    return d0+N@u,V@V.T


class SparseMarginalizationTests(unittest.TestCase):
    def test_symmetric_indefinite_native_solve_multiple_RHS_no_perturbation(self):
        from estimator.temporal_cudss import CudaSparseFactor
        if not torch.cuda.is_available():raise RuntimeError('Actual CUDA is required')
        A=torch.tensor([[1.,2.,.2],[.3,-.5,1.]],device='cuda',dtype=torch.float64)
        K=torch.cat((torch.cat((torch.eye(2,device='cuda',dtype=torch.float64),A),1),
                     torch.cat((A.T,-torch.eye(3,device='cuda',dtype=torch.float64)),1)),0)
        rhs=torch.arange(15,device='cuda',dtype=torch.float64).reshape(5,3)/10
        expected=np.linalg.solve(K.cpu().numpy(),rhs.cpu().numpy())
        with CudaSparseFactor(K.to_sparse_csr()) as factor:
            actual=factor.solve(rhs)
            np.testing.assert_allclose(actual.cpu(),expected,atol=2e-13,rtol=2e-13)
            self.assertEqual(factor.profile['algorithm'],'symmetric_nd')
            self.assertEqual(factor.profile['perturbed_pivots'],0)
            self.assertIsNone(factor.profile['max_LU_nnz_capacity'])
        K[0,1]=.1
        with self.assertRaisesRegex(SparseNumericalFailure,'nonsymmetric'):CudaSparseFactor(K.to_sparse_csr())

    def test_GPU_MAP_marginal_and_source_cross_match_independent_QR(self):
        if not torch.cuda.is_available():raise RuntimeError('CUDA numerical acceptance must be executed on the actual GPU')
        for ideal in [False,True]:
            graph=fixture_graph('cuda',ideal);problem=graph.assemble_sparse_linearized_problem_gpu()
            _,A,r,C,c,_,_=dense_oracle_for(graph)
            mean,P=independent_qr(A.numpy(),r.numpy(),C.numpy(),c.numpy())
            with SparseJointFactorization(problem) as factor:
                step,certificate=factor.solve_constrained_sparse_joint_gpu()
                n=problem.scalar_dimension;device='cuda'
                # Include ALL coordinates in one small oracle query to check every
                # source/state/pixel cross block, not just a conditional diagonal.
                d=torch.arange(n,device=device)
                query=sparse_coo(torch.stack((d,d)),torch.ones(n,device=device,dtype=torch.float64),(n,n))
                full=factor.extract_joint_selected_marginals_gpu(query,block_width=n,rhs_batch=8)[0].cpu().numpy()
                mean_abs=float(np.max(np.abs(step.cpu().numpy()-mean)));cov_abs=float(np.max(np.abs(full-P)))
                self.assertLess(mean_abs,5e-12);self.assertLess(cov_abs,5e-12)
                MARGINAL_WITNESSES.append({'ideal':ideal,'MAP_abs':mean_abs,'MAP_relative':mean_abs/max(np.max(np.abs(mean)),1e-300),
                    'joint_covariance_abs':cov_abs,'joint_covariance_relative':cov_abs/np.max(np.abs(P)),
                    'certificate':certificate,'profile':factor.profile})

    def test_full_schur_with_fill_and_surviving_hard_equality(self):
        if not torch.cuda.is_available():raise RuntimeError('CUDA sparse FEJ acceptance requires actual GPU execution')
        device='cuda';dtype=torch.float64
        A=torch.tensor([[2.,0.,0.],[0.,1.,0.],[0.,0.,1.],[1.,.4,-.2]],device=device,dtype=dtype)
        r=torch.tensor([.3,-.2,.4,.1],device=device,dtype=dtype)
        C=torch.tensor([[1.,-1.,0.],[1.,0.,-1.]],device=device,dtype=dtype)
        c=torch.tensor([.1,-.2],device=device,dtype=dtype)
        problem=SparseLinearizedProblem(A.to_sparse(),r,C.to_sparse(),c,3,{})
        mean,P=independent_qr(A.cpu(),r.cpu(),C.cpu(),c.cpu())
        boundary=marginalize_lag_boundary_sparse_gpu(problem,torch.tensor([0],device=device),torch.zeros(3,device=device,dtype=dtype))
        self.assertEqual(boundary.constraints.shape,(1,2))
        empty=sparse_coo(torch.empty((2,0),device=device,dtype=torch.long),torch.empty(0,device=device,dtype=dtype),(0,2))
        reduced=SparseLinearizedProblem(empty,torch.empty(0,device=device,dtype=dtype),boundary.constraints,boundary.constraint_residual,2,{},
            information=boundary.information,information_gradient=boundary.gradient,constant_cost=boundary.constant)
        with SparseJointFactorization(reduced) as factor:
            step,certificate=factor.solve_constrained_sparse_joint_gpu()
            query=torch.eye(2,device=device,dtype=dtype).to_sparse()
            covariance=factor.extract_joint_selected_marginals_gpu(query,2)[0]
            np.testing.assert_allclose(step.cpu(),mean[1:],atol=3e-13,rtol=3e-13)
            np.testing.assert_allclose(covariance.cpu(),P[1:,1:],atol=3e-13,rtol=3e-13)
            original_cost=float((A.cpu().numpy()@mean+r.cpu().numpy())@(A.cpu().numpy()@mean+r.cpu().numpy())/2)
            self.assertAlmostEqual(certificate['objective'],original_cost,delta=4e-13)
        self.assertGreater(abs(float(boundary.information.to_dense()[0,1])),0.)
        MARGINAL_WITNESSES.append({'surviving_equality':True,'Schur_profile':boundary.profile,
            'posterior_abs':float(np.max(np.abs(covariance.cpu().numpy()-P[1:,1:]))),'objective_abs':abs(certificate['objective']-original_cost)})

    def test_temporal_source_boundary_matches_dense_schur_current_cross_covariance(self):
        if not torch.cuda.is_available():raise RuntimeError('CUDA required for sparse FEJ test')
        graph=fixture_graph('cuda',True);problem=graph.assemble_sparse_linearized_problem_gpu()
        _,A,r,C,c,_,_=dense_oracle_for(graph)
        mean,P=independent_qr(A.numpy(),r.numpy(),C.numpy(),c.numpy())
        expired=torch.cat((torch.arange(5*graph.N,device='cuda'),
             graph.pixel_columns(graph.pixels.keys[graph.pixels.keys//(graph.pixels.cameras*graph.pixels.pixels)==0]).reshape(-1)))
        boundary=marginalize_lag_boundary_sparse_gpu(problem,expired,torch.zeros(problem.scalar_dimension,device='cuda',dtype=torch.float64))
        n=len(boundary.retained_columns)
        empty=sparse_coo(torch.empty((2,0),device='cuda',dtype=torch.long),torch.empty(0,device='cuda',dtype=torch.float64),(0,n))
        p=SparseLinearizedProblem(empty,torch.empty(0,device='cuda',dtype=torch.float64),boundary.constraints,boundary.constraint_residual,n,{},
             information=boundary.information,information_gradient=boundary.gradient,constant_cost=boundary.constant)
        ix=boundary.retained_columns.cpu().numpy()
        with SparseJointFactorization(p) as factor:
            step,certificate=factor.solve_constrained_sparse_joint_gpu()
            result=factor.extract_joint_selected_marginals_gpu(torch.eye(n,device='cuda',dtype=torch.float64).to_sparse(),n)[0].cpu().numpy()
            np.testing.assert_allclose(step.cpu(),mean[ix],atol=2e-10,rtol=2e-10)
            np.testing.assert_allclose(result,P[np.ix_(ix,ix)],atol=2e-10,rtol=2e-10)
        MARGINAL_WITNESSES.append({'source_marginalized':True,'mean_abs':float(np.max(np.abs(step.cpu().numpy()-mean[ix]))),
            'cov_abs':float(np.max(np.abs(result-P[np.ix_(ix,ix)]))),'Schur_profile':boundary.profile})

    def test_conflicting_equalities_and_unobservable_system_fail(self):
        if not torch.cuda.is_available():raise RuntimeError('CUDA required for sparse rank certificate test')
        for case in ['inconsistent','unobservable']:
            A=torch.tensor([[1.,0.],[0.,1.]] if case=='inconsistent' else [[1.,0.]],device='cuda',dtype=torch.float64)
            C=torch.tensor([[1.,0.],[1.,0.]],device='cuda',dtype=torch.float64) if case=='inconsistent' else torch.empty((0,2),device='cuda',dtype=torch.float64)
            p=SparseLinearizedProblem(A.to_sparse(),torch.zeros(len(A),device='cuda',dtype=torch.float64),C.to_sparse(),
                torch.tensor([0.,1.],device='cuda',dtype=torch.float64) if len(C) else torch.empty(0,device='cuda',dtype=torch.float64),2,{})
            with self.assertRaises((SparseNumericalFailure,RuntimeError)):
                with SparseJointFactorization(p) as factor:factor.solve_constrained_sparse_joint_gpu()


if __name__=='__main__':unittest.main()
