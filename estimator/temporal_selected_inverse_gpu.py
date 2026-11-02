"""Sparse GPU LDL and Takahashi inverse entries in double-double arithmetic; CPU symbolic ordering only.

Query cliques are symbolic zero padding, never information/constraint rows.
No global inverse, dense connected component, covariance truncation or probing.

Why double-double: an UNPIVOTED LDL^T of the quasi-definite KKT [[I, A], [A^T, -H]] exists for every symmetric
ordering, but with whitened Jacobians ||A|| ~ 1e6 its element growth is ~||A||^2. In float64 that measurably lost
up to 1e-4 correlation units on real USB/Blade blocks against extended-precision refined references, for METIS and
for measurement-first CAMD orderings alike. The SAME recurrences evaluated with error-free transformations
(two-sum, FMA two-product; ~106-bit significands) reduce that loss by ~2^53. The selected entries are rounded to
float64 only at the very end.

Storage (exact layout, identical recurrences): L (hi, lo) is factored IN PLACE over the values of K (a
level-scheduled entry reads only entries of strictly earlier elimination-tree levels and its own initial value);
work items find their column by binary search in the per-level prefix `off`; the row structure is released after
the factorization; Z overwrites L level by level from the root (Takahashi's column j reads only its own L column and
Z of its ancestors), staged through a per-level buffer so no column reads a slot another program rewrites.
"""
import ctypes as ct
import hashlib
from pathlib import Path
import subprocess
import time
import numpy as np
import torch
import triton as tr
import triton.language as tl
from estimator.temporal_cudss import SparseNumericalFailure,SparseResourceLimit

# All double-double kernels are launched with enable_fp_fusion=False: an implicit a*b+c -> fma contraction makes
# a product rounded in one place and fused in another, silently destroying the two-sum error terms.
CHUNK_MAX=512    # per-program chunk width; rows/columns longer than this are walked in chunks
BYTES_PER_FACTOR_ENTRY=28   # factor phase: L hi/lo 16 + ri 4 + rc 4 + rx 4 (inverse phase: 20 + level stage)
OPERATIONAL_HEADROOM_BYTES=256*1024**2   # allocator/runtime headroom, not a statistical quantity


@tr.jit
def _two_sum(a,b):
    s=a+b;bb=s-a
    return s,(a-(s-bb))+(b-bb)


@tr.jit
def _fast_two_sum(a,b):
    s=a+b
    return s,b-(s-a)


@tr.jit
def _dd_add(ah,al,bh,bl):
    s,e=_two_sum(ah,bh);t,f=_two_sum(al,bl)
    e=e+t;s,e=_fast_two_sum(s,e);e=e+f
    return _fast_two_sum(s,e)


@tr.jit
def _dd_mul(ah,al,bh,bl):
    p=ah*bh;e=tl.fma(ah,bh,-p);e=e+(ah*bl+al*bh)
    return _fast_two_sum(p,e)


@tr.jit
def _dd_div(ah,al,bh,bl):
    q1=ah/bh
    ph,pl=_dd_mul(q1,q1*0.,bh,bl);rh,rl=_dd_add(ah,al,-ph,-pl)
    q2=rh/bh
    ph,pl=_dd_mul(q2,q2*0.,bh,bl);rh,rl=_dd_add(rh,rl,-ph,-pl)
    q3=rh/bh
    s,e=_fast_two_sum(q1,q2)
    return _dd_add(s,e,q3,q3*0.)


@tr.jit
def _dd_sum(h,l):
    return tl.reduce((h,l),0,_dd_add)


@tr.jit
def _find(P,I,column,target,valid):
    lo=tl.load(P+column);hi=tl.load(P+column+1)
    end=hi
    while tl.sum((lo<hi).to(tl.int32),0)>0:
        active=valid&(lo<hi);middle=(lo+hi)//2
        value=tl.load(I+middle,active,2147483647)
        left=value<target
        lo=tl.where(active&left,middle+1,lo);hi=tl.where(active&~left,middle,hi)
        lo=tl.where(valid,lo,end);hi=tl.where(valid,hi,end)
    return lo


@tr.jit
def _work_item(lc,off,lstart,lend,wstart,cp):
    """Global work index -> (column j, factor position, stage index) with off[t] <= w < off[t+1] inside the level."""
    w=wstart+tl.program_id(0).to(tl.int64)
    lo=lstart.to(tl.int64);hi=lend.to(tl.int64)
    while hi-lo>1:
        middle=(lo+hi)//2
        if tl.load(off+middle)<=w:lo=middle
        else:hi=middle
    j=tl.load(lc+lo)
    return j,tl.load(cp+j)+1+(w-tl.load(off+lo)),w-wstart


@tr.jit
def _diag(jobs,start,cp,rp,rc,rx,Lh,Ll,Dh,Dl,flag,B:tl.constexpr):
    j=tl.load(jobs+start+tl.program_id(0));c=tl.load(cp+j)
    ah=tl.load(Lh+c)*0.;al=ah*0.
    for base in range(tl.load(rp+j),tl.load(rp+j+1),B):   # fixed-width chunks: bounded registers for any row length
        p=base+tl.arange(0,B);valid=p<tl.load(rp+j+1)
        col=tl.load(rc+p,valid,0);valid&=col<j
        index=tl.load(rx+p,valid,0);vh=tl.load(Lh+index,valid,0.);vl=tl.load(Ll+index,valid,0.)
        th,tl_=_dd_mul(vh,vl,vh,vl);th,tl_=_dd_mul(th,tl_,tl.load(Dh+col,valid,0.),tl.load(Dl+col,valid,0.))
        sh,sl=_dd_sum(th,tl_);ah,al=_dd_add(ah,al,sh,sl)
    dh,dl=_dd_add(tl.load(Lh+c),tl.load(Ll+c),-ah,-al)
    if (dh==0.) | (tl.abs(dh)==float('inf')) | (dh!=dh):tl.atomic_or(flag,1)
    tl.store(Dh+j,dh);tl.store(Dl+j,dl);tl.store(Lh+c,1.);tl.store(Ll+c,0.)


@tr.jit(do_not_specialize=['lstart','lend','wstart'])
def _column(lc,off,lstart,lend,wstart,ri,cp,rp,rc,rx,Lh,Ll,Dh,Dl,B:tl.constexpr):
    j,position,w=_work_item(lc,off,lstart,lend,wstart,cp);i=tl.load(ri+position)
    ah=tl.load(Lh+position)*0.;al=ah*0.
    for base in range(tl.load(rp+j),tl.load(rp+j+1),B):
        p=base+tl.arange(0,B);valid=p<tl.load(rp+j+1)
        col=tl.load(rc+p,valid,0);valid&=col<j
        jindex=tl.load(rx+p,valid,0)
        found=_find(rp,rc,tl.full((B,),i,tl.int32),col,valid)
        valid&=found<tl.load(rp+i+1)
        match=tl.load(rc+found,valid,-1)==col;valid&=match
        ix=tl.load(rx+found,valid,0)
        th,tl_=_dd_mul(tl.load(Lh+jindex,valid,0.),tl.load(Ll+jindex,valid,0.),tl.load(Lh+ix,valid,0.),tl.load(Ll+ix,valid,0.))
        th,tl_=_dd_mul(th,tl_,tl.load(Dh+col,valid,0.),tl.load(Dl+col,valid,0.))
        sh,sl=_dd_sum(th,tl_);ah,al=_dd_add(ah,al,sh,sl)
    nh,nl=_dd_add(tl.load(Lh+position),tl.load(Ll+position),-ah,-al)
    vh,vl=_dd_div(nh,nl,tl.load(Dh+j),tl.load(Dl+j))
    tl.store(Lh+position,vh);tl.store(Ll+position,vl)


@tr.jit(do_not_specialize=['lstart','lend','wstart'])
def _inverse_column(lc,off,lstart,lend,wstart,ri,cp,Lh,Ll,Th,Tl,flag,B:tl.constexpr):
    j,position,w=_work_item(lc,off,lstart,lend,wstart,cp);i=tl.load(ri+position)
    ah=tl.load(Lh+position)*0.;al=ah*0.
    for base in range(tl.load(cp+j)+1,tl.load(cp+j+1),B):
        p=base+tl.arange(0,B);valid=p<tl.load(cp+j+1)
        k=tl.load(ri+p,valid,0)
        column=tl.minimum(k,i);row=tl.maximum(k,i)
        found=_find(cp,ri,column,row,valid);ok=found<tl.load(cp+column+1)
        matched=tl.load(ri+found,valid&ok,-1)==row;ok&=matched
        if tl.sum((valid&~ok).to(tl.int32),0)>0:tl.atomic_or(flag,2)
        # own column still holds L; ancestors' slots already hold Z
        th,tl_=_dd_mul(tl.load(Lh+p,valid,0.),tl.load(Ll+p,valid,0.),tl.load(Lh+found,valid&ok,0.),tl.load(Ll+found,valid&ok,0.))
        sh,sl=_dd_sum(th,tl_);ah,al=_dd_add(ah,al,sh,sl)
    tl.store(Th+w,-ah);tl.store(Tl+w,-al)


@tr.jit(do_not_specialize=['lstart','wstart'])
def _inverse_diag(lc,off,lstart,wstart,cp,Lh,Ll,Dh,Dl,Th,Tl,B:tl.constexpr):
    t=lstart+tl.program_id(0);j=tl.load(lc+t);c=tl.load(cp+j)
    first=c+1;stage=tl.load(off+t)-wstart
    ah=tl.load(Dh+j)*0.;al=ah*0.
    for base in range(first,tl.load(cp+j+1),B):
        p=base+tl.arange(0,B);valid=p<tl.load(cp+j+1)
        s=stage+(p-first)
        th,tl_=_dd_mul(tl.load(Lh+p,valid,0.),tl.load(Ll+p,valid,0.),tl.load(Th+s,valid,0.),tl.load(Tl+s,valid,0.))
        sh,sl=_dd_sum(th,tl_);ah,al=_dd_add(ah,al,sh,sl)
    dh=tl.load(Dh+j)
    ih,il=_dd_div(dh*0.+1.,dh*0.,dh,tl.load(Dl+j))
    zh,zl=_dd_add(ih,il,-ah,-al)
    tl.store(Lh+c,zh);tl.store(Ll+c,zl)


@tr.jit(do_not_specialize=['lstart','lend','wstart'])
def _scatter(lc,off,lstart,lend,wstart,cp,Lh,Ll,Th,Tl):
    j,position,w=_work_item(lc,off,lstart,lend,wstart,cp)
    tl.store(Lh+position,tl.load(Th+w));tl.store(Ll+position,tl.load(Tl+w))


@tr.jit
def _query(qp,qi,qv,invperm,m,cp,ri,Zh,Zl,out,flag,B:tl.constexpr,W:tl.constexpr):
    point=tl.program_id(0);entry=tl.program_id(1);a=entry//W;b=entry%W
    ka=tl.load(qp+point*W+a)+tl.arange(0,B);kb=tl.load(qp+point*W+b)+tl.arange(0,B)
    va=ka<tl.load(qp+point*W+a+1);vb=kb<tl.load(qp+point*W+b+1)
    ia=tl.load(qi+ka,va,0)+m;ib=tl.load(qi+kb,vb,0)+m
    ia=tl.load(invperm+ia);ib=tl.load(invperm+ib)
    aa=tl.load(qv+ka,va,0.);bb=tl.load(qv+kb,vb,0.)
    col=tl.minimum(ia[:,None],ib[None,:]);row=tl.maximum(ia[:,None],ib[None,:])
    valid=va[:,None]&vb[None,:]
    # Flatten to a vector for sparse binary search and reduction.
    col=tl.reshape(col,(B*B,));row=tl.reshape(row,(B*B,));valid=tl.reshape(valid,(B*B,))
    found=_find(cp,ri,col,row,valid);ok=found<tl.load(cp+col+1)
    ok&=tl.load(ri+found,valid&ok,-1)==row
    if tl.sum((valid&~ok).to(tl.int32),0)>0:tl.atomic_or(flag,4)
    wa=tl.reshape(aa[:,None]+bb[None,:]*0.,(B*B,));wb=tl.reshape(aa[:,None]*0.+bb[None,:],(B*B,))
    wh,wl=_dd_mul(wa,wa*0.,wb,wb*0.)
    th,tl_=_dd_mul(wh,wl,tl.load(Zh+found,valid&ok,0.),tl.load(Zl+found,valid&ok,0.))
    sh,sl=_dd_sum(th,tl_)
    tl.store(out+point*W*W+entry,-(sh+sl))


SMALL={'perm':0,'cp':1,'rp':3,'lp':5,'lc':6,'off':7,'sizes':8}
LARGE={'ri':2,'rc':4,'rx':9}


def symbolic(K,Q,width,first_block=0):
    """CPU pattern-only CHOLMOD analysis of K plus the zero-valued query cliques; returns host arrays."""
    source=Path(__file__).resolve().parents[1]/'cuda/temporal_selected_symbolic.cpp'
    library=Path('/tmp')/('rgbd_selected_symbolic_'+hashlib.sha256(source.read_bytes()).hexdigest()[:16]+'.so')
    if not library.exists():
        subprocess.run(['g++','-std=c++17','-O3','-shared','-fPIC','-I/usr/include/suitesparse',str(source),'-lcholmod','-o',str(library)],check=True,capture_output=True)
    lib=ct.CDLL(str(library));ptr=ct.POINTER(ct.c_int64)
    lib.rgbd_symbolic.argtypes=[ct.c_int64,ct.c_int64,ptr,ptr,ct.c_int64,ptr,ptr,ct.c_int64];lib.rgbd_symbolic.restype=ct.c_void_p
    lib.rgbd_symbolic_size.argtypes=[ct.c_void_p,ct.c_int];lib.rgbd_symbolic_size.restype=ct.c_int64
    lib.rgbd_symbolic_array.argtypes=[ct.c_void_p,ct.c_int];lib.rgbd_symbolic_array.restype=ptr
    lib.rgbd_symbolic_array32.argtypes=[ct.c_void_p,ct.c_int];lib.rgbd_symbolic_array32.restype=ct.POINTER(ct.c_int32)
    lib.rgbd_symbolic_release32.argtypes=[ct.c_void_p,ct.c_int];lib.rgbd_symbolic_free.argtypes=[ct.c_void_p];lib.rgbd_symbolic_error.restype=ct.c_char_p
    k=K.to_sparse_coo().coalesce();index=k.indices();take=index[0]<=index[1]
    row,col=[np.ascontiguousarray(a.cpu().numpy(),dtype=np.int64) for a in index[:,take]]
    del k,index,take
    qi=Q.indices();n=K.shape[0];pair=torch.unique((qi[0]//width)*n+qi[1])
    block=(pair//n).cpu().numpy();support=np.ascontiguousarray((pair%n).cpu().numpy(),dtype=np.int64)
    count=np.bincount(block,minlength=Q.shape[0]//width);qp=np.r_[0,np.cumsum(count)].astype(np.int64)
    result=lib.rgbd_symbolic(n,len(row),row.ctypes.data_as(ptr),col.ctypes.data_as(ptr),len(count),qp.ctypes.data_as(ptr),support.ctypes.data_as(ptr),int(first_block))
    del row,col
    if not result:raise SparseNumericalFailure(lib.rgbd_symbolic_error().decode())
    try:
        out={name:np.ctypeslib.as_array(lib.rgbd_symbolic_array(result,key),shape=(lib.rgbd_symbolic_size(result,key),)).copy() for name,key in SMALL.items()}
        for name,key in LARGE.items():   # copy then release one large array at a time (bounded host peak)
            out[name]=np.ctypeslib.as_array(lib.rgbd_symbolic_array32(result,key),shape=(lib.rgbd_symbolic_size(result,key),)).copy()
            lib.rgbd_symbolic_release32(result,key)
        return out
    finally:lib.rgbd_symbolic_free(result)


def required_bytes(nnz,n,stage=0):
    return nnz*BYTES_PER_FACTOR_ENTRY+n*64+16*stage


def stage_entries(S):
    lp,off=S['lp'],S['off']
    return int(max(off[lp[1:]]-off[lp[:-1]])) if len(lp)>1 else 0


def selected_inverse_blocks_gpu(K,Q,width=6,expected_inertia=None,symbolic_arrays=None,first_block=0,device=None):
    """Q is expressed in FULL KKT coordinates; return -Q K^-1 Q^T blocks (double-double evaluation).

    Static (unpivoted) LDL^T exists for every symmetric ordering of a quasi-definite KKT; the counted signs of D
    are compared with expected_inertia (Sylvester), so an indefinite/singular case is a failure, not a result.
    symbolic_arrays: a symbolic() result for exactly this K and Q (lets the caller size memory first).
    device: where the factor lives; K and Q may stay in host memory (only K's sorted lower-triangle keys/values are
    uploaded, Q only for the final extraction) so that the factor can use the device memory they would occupy."""
    start=time.perf_counter()
    S=symbolic(K,Q,width,first_block) if symbolic_arrays is None else symbolic_arrays;symbolic_seconds=time.perf_counter()-start
    n=len(S['perm']);nnz=len(S['ri']);lp,sizes,hoff=S['lp'],S['sizes'],S['off'];stage=stage_entries(S)
    free=torch.cuda.mem_get_info()[0];estimated=required_bytes(nnz,n,stage)
    if estimated+OPERATIONAL_HEADROOM_BYTES>free:
        raise SparseResourceLimit(f'Exact double-double selected inversion needs {estimated} B (+{OPERATIONAL_HEADROOM_BYTES} B headroom); free VRAM {free} B; '
                                  f'torch allocated {torch.cuda.memory_allocated()} B, reserved {torch.cuda.memory_reserved()} B; nnz_L={nnz}')
    dev=K.device if device is None else torch.device(device);torch.cuda.reset_peak_memory_stats(dev)
    as32=lambda a:torch.as_tensor(np.ascontiguousarray(a,dtype=np.int32),device=dev)
    permutation=torch.as_tensor(S['perm'],device=dev);inverse=torch.empty_like(permutation);inverse[permutation]=torch.arange(n,device=dev)
    cp,lc,off=as32(S['cp']),as32(S['lc']),as32(hoff);ri=as32(S['ri'])
    # Initial values of K (permuted lower triangle) written directly into L's hi slots, chunk by chunk.
    k=K.to_sparse_coo().coalesce();ix=inverse.to(k.device)[k.indices()];take=ix[0]>=ix[1]
    key=ix[1,take]*n+ix[0,take];order=torch.argsort(key);key=key[order].to(dev);value=k.values()[take][order].to(dev)
    del k,ix,take,order,permutation
    Lh=torch.zeros(nnz,device=dev,dtype=torch.float64);placed=0
    for begin in range(0,nnz,1<<22):
        stop=min(nnz,begin+(1<<22));pos=torch.arange(begin,stop,device=dev)
        column=torch.searchsorted(cp,pos.to(torch.int32),right=True).long()-1
        target=column*n+ri[begin:stop].long()
        at=torch.searchsorted(key,target);valid=at<len(key);safe=at.clamp_max(len(key)-1);valid&=key[safe]==target
        Lh[begin:stop]=torch.where(valid,value[safe],0.);placed+=int(valid.sum())
        del pos,column,target,at,valid,safe
    if placed!=len(key):raise SparseNumericalFailure(f'{len(key)-placed} KKT entries fall outside the symbolic factor pattern')
    del key,value
    Ll=torch.zeros_like(Lh)
    rp,rc,rx=as32(S['rp']),as32(S['rc']),as32(S['rx'])
    for name in ('rc','rx'):S[name]=None
    Dh=torch.empty(n,device=dev,dtype=torch.float64);Dl=torch.empty_like(Dh);flag=torch.zeros((),device=dev,dtype=torch.int32)
    torch.cuda.synchronize();t0=time.perf_counter()
    for level in range(len(lp)-1):
        a,b=int(lp[level]),int(lp[level+1]);count=b-a;work=int(hoff[b]-hoff[a]);B=min(CHUNK_MAX,tr.next_power_of_2(int(sizes[2*level])))
        if count:_diag[(count,)](lc,a,cp,rp,rc,rx,Lh,Ll,Dh,Dl,flag,B,enable_fp_fusion=False)
        if work:_column[(work,)](lc,off,a,b,int(hoff[a]),ri,cp,rp,rc,rx,Lh,Ll,Dh,Dl,B,enable_fp_fusion=False)
    torch.cuda.synchronize();factor_seconds=time.perf_counter()-t0
    if int(flag):raise SparseNumericalFailure('Sparse static LDL zero/nonfinite pivot; no perturbation applied')
    inertia=[int((Dh>0).sum()),int((Dh<0).sum())]
    if expected_inertia is not None and tuple(inertia)!=tuple(expected_inertia):
        raise SparseNumericalFailure(f'Static LDL inertia {inertia} differs from the KKT inertia {list(expected_inertia)}')
    minimum_pivot=float(Dh.abs().min())
    del rp,rc,rx;torch.cuda.empty_cache()
    Th=torch.zeros(max(1,stage),device=dev,dtype=torch.float64);Tl=torch.zeros_like(Th);t0=time.perf_counter()
    for level in range(len(lp)-2,-1,-1):
        a,b=int(lp[level]),int(lp[level+1]);count=b-a;work=int(hoff[b]-hoff[a]);B=min(CHUNK_MAX,tr.next_power_of_2(int(sizes[2*level+1])))
        if work:_inverse_column[(work,)](lc,off,a,b,int(hoff[a]),ri,cp,Lh,Ll,Th,Tl,flag,B,enable_fp_fusion=False)
        if count:_inverse_diag[(count,)](lc,off,a,int(hoff[a]),cp,Lh,Ll,Dh,Dl,Th,Tl,B,enable_fp_fusion=False)
        if work:_scatter[(work,)](lc,off,a,b,int(hoff[a]),cp,Lh,Ll,Th,Tl,enable_fp_fusion=False)
    torch.cuda.synchronize();inverse_seconds=time.perf_counter()-t0
    if int(flag):raise SparseNumericalFailure('Selected inverse symbolic pattern is not closed')
    del Th,Tl,Dh,Dl
    qr=Q.to(dev).to_sparse_csr();qwidth=int((qr.crow_indices()[1:]-qr.crow_indices()[:-1]).max())
    out=torch.empty((Q.shape[0]//width,width,width),device=dev,dtype=torch.float64)
    _query[(len(out),width*width)](qr.crow_indices(),qr.col_indices(),qr.values(),inverse,0,cp,ri,Lh,Ll,out,flag,tr.next_power_of_2(max(1,qwidth)),width,enable_fp_fusion=False)
    torch.cuda.synchronize()
    if int(flag) or not bool(torch.isfinite(out).all()):raise SparseNumericalFailure('Unavailable/nonfinite selected query entry')
    asym=float((out-out.transpose(1,2)).abs().max())
    return (out+out.transpose(1,2))/2,{'method':'GPU sparse LDL/Takahashi in double-double (error-free transformations); CPU pattern-only CHOLMOD',
        'arithmetic':'double-double (two-sum, FMA two-product), rounded to float64 at output',
        'symbolic_seconds':symbolic_seconds,'factor_seconds':factor_seconds,'inverse_seconds':inverse_seconds,
        'total_seconds':time.perf_counter()-start,'nnz_L':nnz,'levels':len(lp)-1,'query_blocks':len(out),
        'ordering':'CHOLMOD METIS on KKT + query cliques' if not first_block else 'CHOLMOD CAMD, measurement rows first',
        'LDL_inertia':inertia,'expected_inertia':None if expected_inertia is None else list(expected_inertia),
        'workspace_bytes_estimated':estimated,'level_stage_entries':stage,'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated(dev),
        'minimum_abs_pivot':minimum_pivot,'symmetry_absolute':asym,'dense_global_matrices':False}
