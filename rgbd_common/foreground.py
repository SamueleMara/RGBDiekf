"""V2 M1--M5: exact masked observations and discrete point feasibility.

Background RGB/depth never participate, including in occlusion decisions.
Negative mask support is conclusive only under a declared object-only premise.
"""
import numpy as np
import torch
from .projection_cuda import transport_and_project,build_pixel_bins_and_zbuffer
from .backends.torch_cuda import numpy

UNKNOWN=0
VISIBLE=1
OCCLUDED=2
HARD_MASK_MODEL_VERSION=2


def compact_valid_fg(mask,raw_valid,depth):
    """Sorted native raw IDs, CPU/Torch-CUDA identical M1 domain."""
    if mask.shape!=depth.shape or raw_valid.shape!=depth.shape:raise ValueError('RGB-D mask/raw-validity grid mismatch')
    used=mask.bool()&raw_valid.bool()&torch.isfinite(depth)&(depth>0)
    return torch.nonzero(used.reshape(-1),as_tuple=True)[0]


def exact_binary_mask(values,source='foreground mask'):
    values=np.asarray(values)
    if values.ndim!=2:raise ValueError(f'{source}: expected a two-dimensional binary foreground mask')
    unique=np.unique(values)
    if not (np.isin(unique,[0,1]).all() or np.isin(unique,[0,255]).all()):
        raise ValueError(f'{source}: NON_BINARY_FOREGROUND_MASK values {unique[:20].tolist()}; expected {{0,1}} or {{0,255}}, no thresholding')
    return values!=0


def foreground_gray(frame):
    import cv2
    rgb=np.zeros((*frame.depth.shape,3),np.uint8)
    valid=frame.valid;rgb[valid]=np.asarray(frame.rgb[valid]*255,dtype=np.uint8)
    return cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY)


def foreground_depth_status(predicted,observed,matched):
    """M3: likelihood assignment wins, otherwise nearer FG explains hiding.

    `matched` is the SAME normalized raw-factor vs unassigned comparison used
    by association. Depth sign alone does not reject a noisy consistent match.
    """
    valid=torch.isfinite(observed)&(observed>0)&torch.isfinite(predicted)&(predicted>0)
    result=torch.zeros_like(predicted,dtype=torch.uint8)
    result[valid]=VISIBLE
    result[valid&(observed<predicted)&~matched]=OCCLUDED
    return result


def projected_mask_status(projected,bins,frame,config,backend,model=None,state=None):
    from .observation_model import output,covariance,gaussian_log_probability,unassigned_log_probability
    status=torch.zeros(len(projected['depth']),dtype=torch.uint8,device=backend.device)
    status[bins['candidate_ids']]=OCCLUDED
    ids=bins['visible_ids'];pixels=projected['pixel'][ids,1]*frame.camera.width+projected['pixel'][ids,0]
    host=numpy(pixels);mask=backend.tensor(frame.mask.ravel()[host],torch.bool)
    valid=backend.tensor(frame.valid.ravel()[host],torch.bool)
    front=torch.zeros(len(ids),dtype=torch.uint8,device=backend.device)
    # No RGB or metric depth is even gathered outside Omega^D.
    if valid.any():
        selected=ids[valid];p=pixels[valid];target=numpy(p)
        observed=backend.tensor(np.asarray(frame.depth).ravel()[target])
        predicted=projected['depth'][selected].double()
        matched=torch.zeros(len(selected),dtype=torch.bool,device=backend.device)
        if model is not None and state is not None:
            h,_,_=output(state,backend.tensor(model.xyz)[selected],backend.tensor(model.rgb)[selected],backend.tensor(model.group,torch.long)[selected],frame.camera,backend)
            z=torch.cat([torch.stack([p%frame.camera.width,p//frame.camera.width],1).double(),observed[:,None],backend.tensor(frame.rgb.reshape(-1,3)[target])],1)
            logf=gaussian_log_probability(z-h,covariance(config,observed,backend))+np.log1p(-config.sensor['no_return'])+np.log(config.match_prior/max(1,model.count))
            logu=unassigned_log_probability(frame,config,backend,p)+np.log1p(-config.match_prior)
            matched=logf>logu
        front[valid]=foreground_depth_status(predicted,observed,matched)
    premise=getattr(frame,'object_only_no_external_occluders',None)
    if premise is True:front[~mask]=VISIBLE
    status[ids]=front
    violation=ids[(front==VISIBLE)&~mask]
    return {'status':status,'violation_indices':violation,'front_indices':ids,'front_pixels':pixels,
            'object_only_no_external_occluders':premise,'undetermined_negative_support':int(((front==UNKNOWN)&~mask).sum())}


def check_hard_mask_feasibility(model,state,frame,config,backend,projected=None,bins=None,classify_foreground=True):
    if not classify_foreground and getattr(frame,'object_only_no_external_occluders',None) is not True:
        return {'feasible':True,'violation_indices':torch.empty(0,dtype=torch.long,device=backend.device)}
    if projected is None:
        projected=transport_and_project(backend.tensor(model.xyz,backend.input_dtype),backend.tensor(model.group,torch.long),backend.tensor(state.poses,backend.input_dtype),frame.camera)
        bins=build_pixel_bins_and_zbuffer(projected,frame.camera,backend.tensor(model.point_id,torch.long))
    if classify_foreground:result=projected_mask_status(projected,bins,frame,config,backend,model,state)
    else:
        ids=bins['visible_ids'];pixel=projected['pixel'][ids]
        mask=backend.tensor(frame.mask[numpy(pixel[:,1]),numpy(pixel[:,0])],torch.bool)
        result={'violation_indices':ids[~mask]}
    result['feasible']=len(result['violation_indices'])==0
    return result
