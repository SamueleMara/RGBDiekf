"""Observed-sample -> frozen XYZ surface compatibility, WITHOUT ownership.

The covariance of x-y needs an unavailable cross-covariance. For any joint PSD
covariance, Cov(x-y) <= 2(P_x + Sigma_y). This conservative bound avoids claiming
independence from observations already used in a frozen posterior. The analogous
RGB bound is 4 R_rgb. Component chi-square tests remain the authorized 1% policy;
they are not a calibrated joint confidence or proof of physical correspondence.
"""
import time
from dataclasses import dataclass
import numpy as np
import torch
from scipy.spatial import cKDTree
from .projection import project
from .observations import samples


def appearance_terms(observed,stored,covariance):
    """Exact whitened RGB distance = angular term + brightness term.

    Raw RGB cosine is also returned. Zero-color cosine is undefined (NaN),
    whereas the Gaussian color distance remains well defined, including black.
    No cosine of absolute XYZ and no additional weighted color penalty exist.
    """
    root=torch.linalg.cholesky(covariance)
    whitening=torch.linalg.inv(root)
    a=observed@whitening.T;b=stored@whitening.T
    na=torch.linalg.vector_norm(a,dim=-1);nb=torch.linalg.vector_norm(b,dim=-1)
    dot=(a*b).sum(-1);product=na*nb
    cosine=torch.where(product>0,dot/product,torch.nan).clamp(-1,1)
    angular=2*torch.clamp(product-dot,min=0)
    brightness=(na-nb)**2
    # Direct subtraction is more accurate for nearly identical colors; the
    # angular/brightness decomposition is recorded, never counted a second time.
    distance=((a-b)**2).sum(-1)
    raw_norm=torch.linalg.vector_norm(observed,dim=-1)*torch.linalg.vector_norm(stored,dim=-1)
    raw_cosine=torch.where(raw_norm>0,(observed*stored).sum(-1)/raw_norm,torch.nan).clamp(-1,1)
    return distance,raw_cosine,cosine,angular,brightness


@dataclass
class Coverage:
    pixel:np.ndarray
    explained:np.ndarray
    geometry_explained:np.ndarray
    point_id:np.ndarray
    geometry_point_id:np.ndarray
    nearest_distance_m:np.ndarray
    selected_distance_m:np.ndarray
    geometry_selected_distance_m:np.ndarray
    geometric_d2:np.ndarray
    color_d2:np.ndarray
    rgb_cosine:np.ndarray
    nearest_rgb_cosine:np.ndarray
    angular_d2:np.ndarray
    brightness_d2:np.ndarray
    geometry_rgb_cosine:np.ndarray
    geometry_color_d2:np.ndarray
    stats:dict


class SurfaceIndex:
    """One timestamp only; full SOLVED means and 3x3 position marginals."""
    def __init__(self,state,covariance,colors,point_ids):
        start=time.monotonic();self.state=np.asarray(state);self.colors=np.asarray(colors);self.ids=np.asarray(point_ids,dtype=np.int64)
        self.p=np.asarray(covariance)[:,:3,:3]
        self.finite=np.isfinite(self.state[:,:3]).all(1)
        self.rows=np.flatnonzero(self.finite)
        self.tree=cKDTree(self.state[self.rows,:3]) if len(self.rows) else None
        self.valid_covariance=np.zeros(len(self.ids),bool)
        good=np.isfinite(self.p).all((1,2))
        if good.any():
            symmetric=np.isclose(self.p[good],self.p[good].transpose(0,2,1),rtol=1e-12,atol=1e-14).all((1,2))
            values=np.linalg.eigvalsh(self.p[good]);self.valid_covariance[np.flatnonzero(good)]=symmetric&(values[:,0]>=0)
        self.index_seconds=time.monotonic()-start

    def projection(self,frame,config,backend):
        h,j=project(backend.tensor(self.state),frame.camera);h=h.cpu().numpy();j=j[:,:,:3].cpu().numpy()
        finite=np.isfinite(h).all(1);uv=np.zeros((len(h),2),np.int64)
        uv[finite]=np.floor(h[finite,:2]+.5).astype(np.int64)
        w,height=frame.camera.width,frame.camera.height
        inside=finite&(h[:,2]>0)&(uv[:,0]>=0)&(uv[:,0]<w)&(uv[:,1]>=0)&(uv[:,1]<height)
        reason=np.full(len(h),1,np.uint8) # outside / behind / nonfinite
        indices=np.flatnonzero(inside);pixels=uv[indices,1]*w+uv[indices,0]
        valid=frame.valid.ravel()[pixels];reason[indices]=2 # unknown foreground depth support
        indices=indices[valid];pixels=pixels[valid]
        depth=np.asarray(frame.depth).ravel()[pixels]
        variance=2*(np.einsum('ni,nij,nj->n',j[indices,2],self.p[indices],j[indices,2])+config.measurement_covariance(depth)[:,2,2])
        residual=h[indices,2]-depth
        good=(variance>0)&np.isfinite(variance)&(residual**2<=config.gate(1)*variance)
        reason[indices]=np.where(good,0,np.where(residual>0,3,4))
        reason[~self.valid_covariance]=5
        return h,j,reason

    def match(self,frame,config,backend,raw=None):
        start=time.monotonic();raw=samples(frame,config) if raw is None else raw
        m=len(raw.pixel);n=len(self.ids);nan=lambda:np.full(m,np.nan)
        result=Coverage(raw.pixel.copy(),np.zeros(m,bool),np.zeros(m,bool),np.full(m,-1,np.int64),np.full(m,-1,np.int64),nan(),nan(),nan(),nan(),nan(),nan(),nan(),nan(),nan(),nan(),nan(),{})
        stats={'raw_valid_foreground':m,'raw_pixels':int(frame.depth.size),'invalid_foreground':int(frame.mask.sum())-m,'solved':n,'candidate_edges':0,'visible_edges':0,'geometry_edges':0,'ray_edges':0,'depth_edges':0,'appearance_edges':0,'invalid_uncertainty_edges':0,'observations_with_visible_candidate':0,'observations_passing_3d_gate':0,'observations_passing_ray_gate':0,'tail_probability':config.association_tail_probability,'index_seconds':self.index_seconds,'knn_seconds':0.,'projection_seconds':0.}
        if not m or self.tree is None:result.stats=stats;return result
        t=time.monotonic();h,j,visibility=self.projection(frame,config,backend);stats['projection_seconds']=time.monotonic()-t
        t=time.monotonic();distance,near=self.tree.query(raw.xyz,k=min(config.neighbors,len(self.rows)),workers=1)
        if near.ndim==1:near=near[:,None];distance=distance[:,None]
        near=self.rows[near];order=np.lexsort((self.ids[near],distance),axis=1)
        near=np.take_along_axis(near,order,axis=1);distance=np.take_along_axis(distance,order,axis=1)
        stats['knn_seconds']=time.monotonic()-t;result.nearest_distance_m[:]=distance[:,0]
        k=near.shape[1];size=max(1,backend.chunk_size(3*3*8*40)//k)
        rgb_cov=backend.tensor(4*config.color_covariance);rgb_logdet=float(torch.linalg.slogdet(rgb_cov).logabsdet)
        for lo in range(0,m,size):
            hi=min(m,lo+size);pi=near[lo:hi];count=hi-lo
            xyz=backend.tensor(self.state[pi,:3]);p=backend.tensor(self.p[pi]);error=backend.tensor(raw.xyz[lo:hi,None,:])-xyz
            covariance=2*(p+backend.tensor(raw.covariance[lo:hi,None,:,:]))
            root,info=torch.linalg.cholesky_ex(covariance)
            valid_cov=backend.tensor(self.valid_covariance[pi],torch.bool)&(info==0)&torch.isfinite(root).all((-1,-2))
            safe=torch.where(valid_cov[...,None,None],covariance,torch.eye(3,device=backend.device,dtype=torch.float64))
            # Identity blocks only protect the solver for REJECTED candidates;
            # they are never treated as usable covariance or eligible matches.
            d2=(error*torch.linalg.solve(safe,error[...,None])[...,0]).sum(-1)
            logdet=torch.linalg.slogdet(safe).logabsdet
            hh=backend.tensor(h[pi]);jj=backend.tensor(j[pi]);r=backend.tensor(raw.R[lo:hi,None,:,:])
            innovation=2*(jj@p@jj.transpose(-1,-2)+r)
            safe_innovation=torch.where(valid_cov[...,None,None],innovation,torch.eye(3,device=backend.device,dtype=torch.float64))
            image_error=backend.tensor(raw.z[lo:hi,None,:])-hh
            ray=image_error[...,:2];ray_d2=(ray*torch.linalg.solve(safe_innovation[...,:2,:2],ray[...,None])[...,0]).sum(-1)
            depth_d2=image_error[...,2]**2/safe_innovation[...,2,2]
            observed=backend.tensor(raw.rgb[lo:hi,None,:]);stored=backend.tensor(self.colors[pi])
            dc,cosine,_,angular,brightness=appearance_terms(observed,stored,rgb_cov)
            visible=backend.tensor(visibility[pi]==0,torch.bool)&valid_cov
            geometric=visible&torch.isfinite(d2)&(d2<=config.gate(3))
            ray_ok=geometric&torch.isfinite(ray_d2)&(ray_d2<=config.gate(2))
            geometry_ok=ray_ok&torch.isfinite(depth_d2)&(depth_d2<=config.gate(1))
            color_ok=geometry_ok&torch.isfinite(dc)&(dc<=config.gate(3))
            geo_cost=.5*(d2+logdet)
            # Composite marginal Gaussian NLL: normalized physical error terms,
            # no tunable geometry/color weight and no joint-independence claim.
            app_cost=geo_cost+.5*(dc+rgb_logdet)
            stats['candidate_edges']+=pi.size;stats['visible_edges']+=int(visible.sum());stats['geometry_edges']+=int(geometric.sum());stats['ray_edges']+=int(ray_ok.sum());stats['depth_edges']+=int(geometry_ok.sum());stats['appearance_edges']+=int(color_ok.sum());stats['invalid_uncertainty_edges']+=int((~valid_cov).sum())
            stats['observations_with_visible_candidate']+=int(visible.any(1).sum());stats['observations_passing_3d_gate']+=int(geometric.any(1).sum());stats['observations_passing_ray_gate']+=int(ray_ok.any(1).sum())
            result.nearest_rgb_cosine[lo:hi]=cosine[:,0].cpu().numpy()
            rows=torch.arange(count,device=backend.device)
            for app,eligible,cost in [(False,geometry_ok,geo_cost),(True,color_ok,app_cost)]:
                finite_cost=torch.where(eligible,cost,torch.inf);best=finite_cost.argmin(1);good=torch.isfinite(finite_cost[rows,best]).cpu().numpy();col=best.cpu().numpy()
                selected=self.ids[pi[np.arange(count),col]];selected[~good]=-1
                selected_distance=distance[lo:hi][np.arange(count),col].copy();selected_distance[~good]=np.nan
                if not app:
                    result.geometry_explained[lo:hi]=good;result.geometry_point_id[lo:hi]=selected;result.geometry_selected_distance_m[lo:hi]=selected_distance
                    for name,value in [('geometry_rgb_cosine',cosine),('geometry_color_d2',dc)]:
                        x=value[rows,best].cpu().numpy();x[~good]=np.nan;getattr(result,name)[lo:hi]=x
                else:
                    result.explained[lo:hi]=good;result.point_id[lo:hi]=selected;result.selected_distance_m[lo:hi]=selected_distance
                    for name,value in [('geometric_d2',d2),('color_d2',dc),('rgb_cosine',cosine),('angular_d2',angular),('brightness_d2',brightness)]:
                        x=value[rows,best].cpu().numpy();x[~good]=np.nan;getattr(result,name)[lo:hi]=x
        stats['visibility_point_reasons']={name:int((visibility==i).sum()) for i,name in enumerate(('verified','outside_or_behind','unknown_foreground_depth','occluded','in_front_depth_inconsistent','invalid_uncertainty'))}
        for name,mask,selected in [('geometry',result.geometry_explained,result.geometry_point_id),('appearance',result.explained,result.point_id)]:
            _,counts=np.unique(selected[mask],return_counts=True)
            stats[name+'_explained']=int(mask.sum());stats[name+'_unexplained']=int((~mask).sum());stats[name+'_many_to_one_extra_samples']=int((counts-1).sum())
        stats['seconds']=time.monotonic()-start;result.stats=stats
        return result
