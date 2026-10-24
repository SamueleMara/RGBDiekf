"""Closed sparse covariance of free-point grouped corrections.

P = D + G C G^T + G V^T + V G^T (within each computational block).
D is point-block diagonal, G fixed canonical, V has 12 columns per block.
The closure follows from F G = G T, where T has the signed dt coupling.
All cross-point terms survive prediction/update. No dense point-pair matrix
or growing temporal history is required. Grouped mean restriction is not a
claim of unrestricted Bayesian inference or of physical rigidity.
"""
import numpy as np
import torch
from .filter_iekf import inverse_spd
from rgbd_common.backends.torch_cuda import stable_segment_sum


class CovarianceOperator:
    def __init__(self,p0,g,membership,backend,tolerance):
        self.D=backend.tensor(np.asarray(p0).copy());self.g=g;self.backend=backend
        self.member=np.asarray(membership);self.groups=int(self.member.max())+1
        self.V=torch.zeros_like(g);self.C=g.new_zeros((self.groups,12,12))
        self.history=[];self.pending=None;self.tolerance=tolerance;self.last_solve=[]
        self.order=np.argsort(self.member,kind='stable')
        self.bounds=np.r_[0,np.cumsum(np.bincount(self.member,minlength=self.groups))]
        member_tensor=backend.tensor(self.member,torch.long)
        gram=stable_segment_sum(g.transpose(1,2)@g,member_tensor,self.groups)
        self.gram_inverse=torch.linalg.pinv(gram,hermitian=True)

    def prediction(self,F,Q):
        self.pending=(np.asarray(F).copy(),np.asarray(Q).copy())
        f=self.backend.tensor(F);q=self.backend.tensor(Q)
        t=torch.eye(12,device=f.device,dtype=f.dtype);t[:6,6:]=torch.eye(6,device=f.device,dtype=f.dtype)*f[0,3]
        self.D=f@self.D@f.T+q;self.V=f@self.V@t.T;self.C=t@self.C@t.T

    def batches(self):
        for lo in range(0,self.groups,16):
            hi=min(self.groups,lo+16);lengths=np.diff(self.bounds[lo:hi+1]);width=max(4,int(lengths.max()))
            index=np.zeros((hi-lo,width),np.int64);valid=np.arange(width)[None,:]<lengths[:,None]
            for row,group in enumerate(range(lo,hi)):index[row,:lengths[row]]=self.order[self.bounds[group]:self.bounds[group+1]]
            yield lo,hi,index,valid

    def packed(self,lo,hi,index,valid):
        b=self.backend;mask=b.tensor(valid,torch.bool);ix=b.tensor(index,torch.long)
        g=torch.where(mask[:,:,None,None],self.g[ix],0.)
        v=torch.where(mask[:,:,None,None],self.V[ix],0.)
        d=self.D[ix].clone();d[~mask]=torch.eye(6,device=d.device,dtype=d.dtype)
        c=self.C[lo:hi];root=torch.linalg.cholesky(d)
        # Whiten by D, then QR: inversion lives in <=24 dimensions. Identity
        # here is the whitened existing D, never a fabricated noise floor.
        u=torch.cat((g,v),-1);white=torch.linalg.solve_triangular(root,u,upper=False)
        q,r=torch.linalg.qr(white.flatten(1,2),mode='reduced')
        core=d.new_zeros((hi-lo,24,24));eye=torch.eye(12,device=d.device,dtype=d.dtype)
        core[:,:12,:12]=c;core[:,:12,12:]=eye;core[:,12:,:12]=eye
        small=torch.eye(r.shape[1],device=d.device,dtype=d.dtype)+r@core@r.transpose(1,2)
        small=(small+small.transpose(1,2))/2
        chol,info=torch.linalg.cholesky_ex(small)
        if bool((info!=0).any()):raise ArithmeticError('Closed covariance operator is not numerically SPD; no jitter/clipping')
        def apply(rhs):
            gt=(g.transpose(-1,-2)@rhs).sum(1);vt=(v.transpose(-1,-2)@rhs).sum(1)
            return d@rhs+g@(c@gt+vt)[:,None]+v@gt[:,None]
        def solve(rhs):
            w=torch.linalg.solve_triangular(root,rhs,upper=False)
            flat=w.flatten(1,2);coordinate=q.transpose(1,2)@flat
            delta=torch.cholesky_solve(coordinate,chol)-coordinate
            whitened=(flat+q@delta).reshape_as(w)
            return torch.linalg.solve_triangular(root.transpose(-1,-2),whitened,upper=True)
        return ix,mask,g,v,c,d,apply,solve

    def information(self,pminus):
        info=self.g.new_zeros((self.groups,12,12));self.last_solve=[]
        for lo,hi,index,valid in self.batches():
            ix,mask,g,v,c,d,apply,solve=self.packed(lo,hi,index,valid)
            z=solve(g);norm=torch.sqrt((g*g).sum((1,2)));denom=torch.where(norm>0,norm,torch.ones_like(norm))
            for refinement in range(8):
                residual=g-apply(z);relative=torch.sqrt((residual*residual).sum((1,2)))/denom
                if bool((relative<=self.tolerance).all()):break
                z+=solve(residual)
            if bool((relative>self.tolerance).any()):raise ArithmeticError(f'Closed covariance solve residual {float(relative.max())} exceeds {self.tolerance}; no silent inaccurate solve')
            info[lo:hi]=(g.transpose(-1,-2)@z).sum(1)
            self.last_solve.append({'iterations':1,'refinement_steps':refinement,'max_relative_residual':float(relative.max()),'reduced_dimension':24})
        return (info+info.transpose(1,2))/2

    def update(self,pminus,measurement,m):
        posterior=pminus.clone();newV=torch.zeros_like(self.V);newC=torch.zeros_like(self.C)
        for lo,hi,index,valid in self.batches():
            ix,mask,g,v,c,d,apply,_=self.packed(lo,hi,index,valid)
            a=torch.where(mask[:,:,None,None],measurement[ix],0.)
            ag=a@g;z=apply(ag);j=(g.transpose(-1,-2)@ag).sum(1)
            w=(g.transpose(-1,-2)@a@z).sum(1);mi=m[lo:hi]
            vv=v-z@mi[:,None];cc=c+mi@(j+w)@mi
            # Keep V orthogonal to G. Otherwise two very large, redundant
            # terms in G C G^T + G V^T + V G^T destroy numerical precision.
            projected=self.gram_inverse[lo:hi]@(g.transpose(-1,-2)@vv).sum(1)
            vv-=g@projected[:,None];cc+=projected+projected.transpose(1,2)
            cc=(cc+cc.transpose(1,2))/2
            cross=g@vv.transpose(-1,-2)
            values=d+g@cc[:,None]@g.transpose(-1,-2)+cross+cross.transpose(-1,-2)
            values=(values+values.transpose(-1,-2))/2
            if bool(((torch.linalg.cholesky_ex(values)[1]!=0)&mask).any()):
                raise ArithmeticError('Closed covariance marginal lost SPD; no artificial repair')
            posterior[ix[mask]]=values[mask];newV[ix[mask]]=vv[mask];newC[lo:hi]=cc
        self.V=newV;self.C=newC;self.history.append({'F':self.pending[0],'Q':self.pending[1]})
        return posterior,{'cross_factor':self.V,'group_core':self.C,'update_reduced_inverse':m,
            'prediction_F':self.backend.tensor(self.pending[0]),'prediction_Q':self.backend.tensor(self.pending[1])}
