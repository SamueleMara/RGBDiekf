"""Complete sparse FEJ Schur boundary, including residual hard constraints.

Symbolic matching is one batched CPU graph operation, timed explicitly. Numeric
factorization, solves and fill assembly run on CUDA. Numerically dependent pivot
rows fail the solve certificate; they are never regularized or silently dropped.
"""
from dataclasses import dataclass
import time
import numpy as np
import torch
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching
from estimator.temporal_cudss import CudaSparseFactor, SparseNumericalFailure, SparseResourceLimit
from estimator.temporal_sparse_gpu import sparse_coo
from estimator.temporal_sparse_solve import connected_columns_gpu


def sparse_slice(matrix, rows, columns):
    rowmap=torch.full((matrix.shape[0],),-1,device=rows.device,dtype=torch.long)
    colmap=torch.full((matrix.shape[1],),-1,device=rows.device,dtype=torch.long)
    rowmap[rows]=torch.arange(len(rows),device=rows.device)
    colmap[columns]=torch.arange(len(columns),device=columns.device)
    ix=matrix.coalesce().indices();values=matrix.coalesce().values()
    selected=(rowmap[ix[0]]>=0)&(colmap[ix[1]]>=0)
    return sparse_coo(torch.stack((rowmap[ix[0,selected]],colmap[ix[1,selected]])),values[selected],(len(rows),len(columns)))


@dataclass
class SparseFEJBoundary:
    information: torch.Tensor
    gradient: torch.Tensor
    constraints: torch.Tensor
    constraint_residual: torch.Tensor
    linearization: torch.Tensor
    constant: float
    retained_columns: torch.Tensor
    profile: dict


def schur_product_gpu(factor, B, workspace_bytes, rhs_batch=8):
    """Compute sparse B K^-1 B^T with every induced component fill edge retained."""
    n=B.shape[1];device=B.device
    empty=sparse_coo(torch.empty((2,0),dtype=torch.long,device=device),torch.empty(0,dtype=torch.float64,device=device),(0,n))
    # K connectivity (not B.T B) determines independent eliminated-variable blocks.
    labels,rounds=connected_columns_gpu(empty,empty,factor.matrix.to_sparse_coo())
    qr,qc=B.indices();value=B.values()
    if B.shape[0]*n>torch.iinfo(torch.int64).max:raise OverflowError('Schur query key overflow')
    pair,inverse=torch.unique(qr*n+labels[qc],sorted=True,return_inverse=True)
    row,component=pair//n,pair%n
    order=torch.argsort(component,stable=True)
    comp,counts=torch.unique_consecutive(component[order],return_counts=True)
    starts=torch.cumsum(counts,0)-counts
    total=int(counts.square().sum())
    # Induced dense fill is real mathematical fill, stored as sparse coordinates.
    required=total*(2*8+2*8)+factor.matrix.shape[0]*rhs_batch*8*3
    if required>workspace_bytes:
        # Exact Schur fill is dense per eliminated component (sum of squared boundary sizes): a measured resource limit.
        error=SparseResourceLimit(f'FULL_SCHUR_FILL_RESOURCE_LIMIT: {total} induced entries require at least {required} bytes; budget={workspace_bytes}; '
                                  f'eliminated components with boundary rows={len(counts)}, largest boundary={int(counts.max())} rows')
        error.profile={'induced_entries':total,'required_bytes':required,'workspace_bytes':workspace_bytes,'boundary_components':len(counts),
                       'largest_boundary_rows':int(counts.max()),'boundary_rows':int(counts.sum()),'boundary_rows_quantiles':torch.quantile(counts.double(),torch.tensor([.5,.9,.99,1.],device=counts.device,dtype=torch.float64)).tolist()}
        raise error
    group=torch.repeat_interleave(torch.arange(len(counts),device=device),counts.square())
    offsets=torch.cumsum(counts.square(),0)-counts.square()
    position=torch.arange(total,device=device)-torch.repeat_interleave(offsets,counts.square())
    left=order[starts[group]+position//counts[group]]
    right=order[starts[group]+position%counts[group]]
    ranks=torch.arange(len(order),device=device)-torch.repeat_interleave(starts,counts)
    color=torch.empty_like(ranks);color[order]=ranks
    pieces=sparse_coo(torch.stack((inverse,qc)),value,(len(pair),n))
    answer=torch.zeros(total,device=device,dtype=torch.float64)
    arithmetic=torch.zeros_like(answer)
    colors=int(counts.max()) if len(counts) else 0
    for begin in range(0,colors,rhs_batch):
        end=min(colors,begin+rhs_batch)
        take=(color[inverse]>=begin)&(color[inverse]<end)
        rhs=sparse_coo(torch.stack((qc[take],color[inverse[take]]-begin)),value[take],(n,end-begin)).to_dense()
        solution=factor.solve(rhs)
        product=torch.sparse.mm(pieces,solution)
        bound=torch.sparse.mm(pieces.abs(),solution.abs())
        wanted=(color[right]>=begin)&(color[right]<end)
        answer[wanted]=product[left[wanted],color[right[wanted]]-begin]
        arithmetic[wanted]=bound[left[wanted],color[right[wanted]]-begin]
    index=torch.stack((row[left],row[right]))
    result=sparse_coo(index,answer,(B.shape[0],B.shape[0]))
    bounds=sparse_coo(index,arithmetic,(B.shape[0],B.shape[0]))
    return result,bounds,{'possible_fill_entries':total,'actual_product_nnz':result._nnz(),
                         'rhs_colors':colors,'elimination_component_count':len(comp),'connectivity_iterations':rounds,
                         'estimated_workspace_bytes':required}


def marginalize_lag_boundary_sparse_gpu(problem, expired_columns, linearization, workspace_bytes=512*1024**2):
    started=time.perf_counter();A,r,C,c=problem.A,problem.residual,problem.C,problem.constraint
    device=A.device;n=A.shape[1]
    if device.type!='cuda':raise ValueError('Sparse FEJ numeric elimination requires CUDA')
    keep=torch.ones(n,device=device,dtype=torch.bool);keep[expired_columns]=False
    retained=torch.where(keep)[0]
    # Forming normal information here is explicit; its accuracy is independently
    # checked against square-root QR, and no weak mode is clipped or discarded.
    H=torch.sparse.mm(A.T,A).coalesce()
    g=torch.sparse.mm(A.T,r[:,None]).flatten()
    constant=float(r@r/2)+problem.constant_cost
    if problem.information is not None:
        H=(H+problem.information).coalesce();g+=problem.information_gradient
    oldC=sparse_slice(C,torch.arange(C.shape[0],device=device),expired_columns)
    symbolic=time.perf_counter()
    indices=oldC.indices().cpu().numpy()
    structure=coo_matrix((np.ones(len(indices[0]),dtype=np.int8),(indices[0],indices[1])),shape=oldC.shape).tocsr()
    matching=maximum_bipartite_matching(structure,perm_type='column')
    independent=torch.as_tensor(np.flatnonzero(matching>=0),device=device,dtype=torch.long)
    remaining=torch.as_tensor(np.flatnonzero(matching<0),device=device,dtype=torch.long)
    symbolic_seconds=time.perf_counter()-symbolic
    ci,cv=C.indices(),C.values()
    K=sparse_coo(torch.cat((H.indices(),torch.stack((ci[1],n+ci[0])),torch.stack((n+ci[0],ci[1]))),1),
                 torch.cat((H.values(),cv,cv)),(n+len(c),n+len(c)))
    eliminated=torch.cat((expired_columns,n+independent))
    boundary=torch.cat((retained,n+remaining))
    Kv=sparse_slice(K,eliminated,eliminated)
    B=sparse_slice(K,boundary,eliminated)
    Kuu=sparse_slice(K,boundary,boundary)
    b=torch.cat((g,c));bv=b[eliminated];bu=b[boundary]
    norm=torch.zeros(Kv.shape[0],device=device,dtype=torch.float64)
    norm.scatter_add_(0,Kv.indices()[0],Kv.values().abs())
    if bool((norm==0).any()):raise SparseNumericalFailure('FULL_FEJ_ZERO_ELIMINATION_ROW')
    scale=norm.rsqrt();ki=Kv.indices()
    Kv=sparse_coo(ki,Kv.values()*scale[ki[0]]*scale[ki[1]],Kv.shape)
    B=sparse_coo(B.indices(),B.values()*scale[B.indices()[1]],B.shape);bv=bv*scale
    with CudaSparseFactor(Kv) as factor:
        product,bounds,profile=schur_product_gpu(factor,B,workspace_bytes)
        S=(Kuu-product).coalesce()
        solved=factor.solve(bv).flatten()
        reduced=bu-torch.sparse.mm(B,solved[:,None]).flatten()
        constant-=float(bv@solved/2)
        profile.update(factor.profile)
    nr=len(retained);S=((S+S.T).coalesce()/2).coalesce()
    ix=S.indices();value=S.values()
    residual_duals=(ix[0]>=nr)&(ix[1]>=nr)
    if bool(residual_duals.any()):
        # These entries vanish algebraically because unmatched old-constraint
        # rows lie in the selected row span. Verify the floating cancellation
        # against an entrywise product error bound, never a geometry/noise gate.
        dualrows=torch.arange(nr,len(boundary),device=device)
        block=sparse_slice(S,dualrows,dualrows)
        bound=sparse_slice(bounds,dualrows,dualrows)
        bi=bound.indices();qi=block.indices()
        keys=bi[0]*len(dualrows)+bi[1];wanted=qi[0]*len(dualrows)+qi[1]
        positions=torch.searchsorted(keys,wanted)
        found=positions<len(keys)
        safe=positions.clamp_max(max(0,len(keys)-1))
        magnitude=torch.zeros_like(block.values())
        if len(keys):magnitude[found]=bound.values()[safe[found]]
        tolerance=64*torch.finfo(torch.float64).eps*max(1,len(eliminated))*magnitude
        if bool((block.values().abs()>tolerance).any()):
            raise SparseNumericalFailure('FULL_FEJ_CONSTRAINT_RANK_FAILURE: symbolic pivot set does not certify remaining hard rows')
    hmask=(ix[0]<nr)&(ix[1]<nr)
    newH=sparse_coo(ix[:,hmask],value[hmask],(nr,nr))
    cmask=(ix[0]>=nr)&(ix[1]<nr)
    newC=sparse_coo(torch.stack((ix[0,cmask]-nr,ix[1,cmask])),value[cmask],(len(remaining),nr))
    profile.update(total_seconds=time.perf_counter()-started,symbolic_matching_seconds=symbolic_seconds,
        eliminated_scalars=len(expired_columns),retained_scalars=nr,old_constraints=len(c),
        eliminated_constraint_rank_candidate=len(independent),remaining_hard_constraints=len(remaining),
        FEJ_information_nnz=newH._nnz(),FEJ_constraint_nnz=newC._nnz(),
        exactness='complete sparse Schur; numerical certification required; normal-information conditioning audited against QR')
    return SparseFEJBoundary(newH,reduced[:nr],newC,reduced[nr:],linearization[retained].clone(),constant,retained,profile)
