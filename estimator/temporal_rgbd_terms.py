"""One shared metric-depth bilinear temporal observation, for both solvers.

Jacobian is d h / d state; graph warp residual has its negative. Hard invalid
supports return finite zero storage and never become factors.
"""
from dataclasses import dataclass
import math
import torch
from estimator.projective_sensor import REASONS
from estimator.temporal_zbuffer import selected_owners

TEMPORAL_REASONS={**REASONS,11:'BOOTSTRAP_PENDING',12:'REFERENCE_OUTSIDE_LAG'}


def bilinear(q, values, cell=None):
    base = torch.floor(q) if cell is None else cell.to(q.dtype)
    u, v = (q-base).unbind(-1)
    w = torch.stack(((1-u)*(1-v),u*(1-v),(1-u)*v,u*v), -1)
    dw = torch.stack((torch.stack((-(1-v),1-v,-v,v),-1),
                      torch.stack((-(1-u),-u,1-u,u),-1)), -1)
    h = torch.einsum('...a,...ad->...d',w,values)
    grad = torch.einsum('...ab,...ad->...db',dw,values)
    return h, grad, w, dw


def temporal_prediction(rig, s_t, s_ref, cam, pixel_uv, clean_values, cell=None):
    ut, zt, jt, jzt = rig.project_rows(s_t[..., :3],cam)
    ur, zr, jr, jzr = rig.project_rows(s_ref[..., :3],cam)
    q = pixel_uv-ut+ur
    h, grad, w, dw = bilinear(q,clean_values,cell)
    h = h.clone(); h[...,3] += zt-zr
    ht = torch.zeros((*h.shape,6), dtype=h.dtype, device=h.device)
    hr = torch.zeros_like(ht)
    ht[...,:3] = -grad@jt; hr[...,:3] = grad@jr
    ht[...,3,:3] += jzt; hr[...,3,:3] -= jzr
    return h, ht, hr, q, w, dw, grad


def raw_values(frame, cam, pixels):
    """Only mask-positive finite valid observations are returned; reject first."""
    yy, xx = pixels//frame.W, pixels%frame.W
    ok = frame.mask[cam,yy,xx] & frame.raw_valid[cam,yy,xx]
    if not bool(ok.all()): raise ValueError('BACKGROUND_OR_INVALID_RAW_PIXEL')
    out = torch.cat((frame.rgb[cam,yy,xx],frame.depth[cam,yy,xx,None]),-1)
    if not bool(torch.isfinite(out).all()) or bool((out[...,3]<=0).any()):
        raise ValueError('NONFINITE_OR_NONPOSITIVE_RAW_PIXEL')
    return out


def sensor_covariance(values, noise):
    v = torch.empty_like(values)
    v[...,:3] = noise.rgb_variance.to(values)
    v[...,3] = noise.depth_sigma2(values[...,3])
    if not bool((v>0).all()) or not bool(torch.isfinite(v).all()):
        raise ValueError('TEMPORAL_SENSOR_COVARIANCE_NOT_SPD')
    return torch.diag_embed(v)


def validate_reference_patch(frame, cam, q, rig, noise, policy):
    M = len(q); base = torch.floor(torch.nan_to_num(q)).long()
    inside = torch.isfinite(q).all(-1) & (base[:,0]>=0) & (base[:,1]>=0) & (base[:,0]+1<frame.W) & (base[:,1]+1<frame.H)
    xy = base[:,None,:] + torch.tensor([[0,0],[1,0],[0,1],[1,1]],device=q.device)
    xy[...,0].clamp_(0,frame.W-1); xy[...,1].clamp_(0,frame.H-1)
    pix = xy[...,1]*frame.W+xy[...,0]
    cc = cam[:,None].expand(-1,4)
    mk = frame.mask[cc,xy[...,1],xy[...,0]].all(1)
    dv = frame.raw_valid[cc,xy[...,1],xy[...,0]].all(1)
    vals = torch.zeros((M,4,4),dtype=q.dtype,device=q.device)
    reason = torch.zeros(M,dtype=torch.uint8,device=q.device)
    reason[~dv] = 4; reason[~mk] = 3; reason[~inside] = 2
    ix = torch.where(reason==0)[0]
    if len(ix):
        vv = torch.cat((frame.rgb[cc[ix],xy[ix,:,1],xy[ix,:,0]],
                        frame.depth[cc[ix],xy[ix,:,1],xy[ix,:,0],None]),-1).to(q.dtype)
        finite = torch.isfinite(vv).all((1,2)) & (vv[:,:,3]>0).all(1)
        reason[ix[~finite]] = 4
        ix, vv = ix[finite], vv[finite]
        vals[ix] = vv
        z = vv[:,:,3]; zm = z.mean(1)
        allowance = policy['same_surface_n_sigma']*math.sqrt(2)*noise.depth_sigma2(zm).sqrt() + zm*math.tan(math.radians(policy['same_surface_max_incidence_deg']))*math.sqrt(2)/rig.K[cam[ix],:2].min(1).values
        reason[ix[(z.max(1).values-z.min(1).values)>allowance]] = 5
    vals[reason!=0] = 0
    return pix, vals, reason, base


@dataclass
class TemporalRows:
    point: torch.Tensor
    camera: torch.Tensor
    pixel: torch.Tensor
    reference_frame: torch.Tensor
    reference_state: torch.Tensor
    reference_pixels: torch.Tensor
    values: torch.Tensor
    y: torch.Tensor
    h: torch.Tensor
    H_t: torch.Tensor
    H_ref: torch.Tensor
    q: torch.Tensor
    weights: torch.Tensor
    cell: torch.Tensor
    R: torch.Tensor

    def __len__(self): return len(self.point)

    @property
    def nonsmooth(self):
        return (self.q==torch.floor(self.q)).any(-1)


def build_temporal_rows(s, owners, frame, references, frames, rig, noise, policy, warp_cov, ordinal, lag):
    """No temporal row on first camera acquisition. Every row has a raw key."""
    c,p,pixel=selected_owners(owners);y,x=pixel//frame.W,pixel%frame.W
    reason = owners.reason.clone()
    mk = frame.mask[c,y,x]; valid = frame.raw_valid[c,y,x] & torch.isfinite(frame.depth[c,y,x]) & (frame.depth[c,y,x]>0) & torch.isfinite(frame.rgb[c,y,x]).all(1)
    reason[c[~mk],p[~mk]] = 3
    reason[c[mk & ~valid],p[mk & ~valid]] = 4
    ok = mk & valid
    rf = references.frame[c,p]
    age = ordinal-references.ordinal[c,p]
    ready = (rf>=0) & (age<lag) & (age>0)
    reason[c[ok & ~ready],p[ok & ~ready]]=11
    stale=ok & (rf>=0) & (age>=lag)
    reason[c[stale],p[stale]]=12
    lifecycle = {'bootstrap': int((ok & ~ready).sum()),'reference_outside_lag': int((ok & (rf>=0) & (age>=lag)).sum()),
                 'occluded_no_row': int((owners.reason==6).sum()), 'invalid_current': int((~ok).sum())}
    c,p,pixel,rf = c[ok & ready],p[ok & ready],pixel[ok & ready],rf[ok & ready]
    ref_s = references.state[c,p]
    uv = torch.stack((pixel%frame.W,pixel//frame.W),-1).to(s.dtype)
    ut = rig.project_rows(s[p,:3],c)[0]; ur = rig.project_rows(ref_s[:,:3],c)[0]
    q = uv-ut+ur
    keys = torch.zeros((len(p),4),dtype=torch.long,device=s.device)
    vals = torch.zeros((len(p),4,4),dtype=s.dtype,device=s.device)
    cell = torch.zeros((len(p),2),dtype=torch.long,device=s.device)
    good = torch.zeros(len(p),dtype=torch.bool,device=s.device)
    for k in rf.unique().tolist():
        ix = torch.where(rf==k)[0]
        pix,v,r,b = validate_reference_patch(frames[int(k)],c[ix],q[ix],rig,noise,policy)
        keys[ix],vals[ix],cell[ix],good[ix] = pix,v,b,r==0
        reason[c[ix[r!=0]],p[ix[r!=0]]] = r[r!=0]
    lifecycle['invalid_reference'] = int((~good).sum())
    p,c,pixel,rf,ref_s,keys,vals,cell,uv = (a[good] for a in (p,c,pixel,rf,ref_s,keys,vals,cell,uv))
    h,ht,hr,q,w,_,_ = temporal_prediction(rig,s[p],ref_s,c,uv,vals,cell)
    yy = raw_values(frame,c,pixel).to(s.dtype)
    R = sensor_covariance(yy,noise) + torch.einsum('ma,maij->mij',w*w,sensor_covariance(vals,noise)) + warp_cov
    return TemporalRows(p,c,pixel,rf,ref_s,keys,vals,yy,h,ht,hr,q,w,cell,R), reason, lifecycle
