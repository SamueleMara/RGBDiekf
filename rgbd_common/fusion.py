"""Conditional precision fusion with immutable raw IDs and shared pose handles."""
import numpy as np
import torch
from .geometry import PointMap,ObservationIDs
from .lie import State
from .matching_cuda import match_rgbd
from .observation_model import covariance
from .backends.torch_cuda import numpy
from .progress import progress
from .projection_cuda import rigid_transform


def _inverse(value,backend=None):
    if backend is None:return np.linalg.inv(value)
    return numpy(torch.linalg.inv(backend.tensor(value)))


def backproject(frame,config,backend,pixels=None):
    if pixels is None:
        from .foreground import compact_valid_fg
        pixels=numpy(compact_valid_fg(backend.tensor(frame.mask,torch.bool),backend.tensor(frame.raw_valid,torch.bool),backend.tensor(frame.depth)))
    pixels=np.asarray(pixels,dtype=np.int64)
    if pixels.ndim!=1 or np.any((pixels<0)|(pixels>=frame.depth.size)):raise ValueError('Backprojection requires in-image raw pixel IDs')
    if not np.all(frame.valid.ravel()[pixels]):raise ValueError('Backprojection source must be finite positive-depth masked foreground')
    if not len(pixels):return np.empty((0,3)),np.empty((0,3)),np.empty((0,6,6)),pixels
    fx,fy,cx,cy=map(float,frame.camera.intrinsics);cw=backend.tensor(frame.camera.world_to_camera);rotation=cw[:3,:3].T
    xyz=[];noise=[];size=backend.chunk_size(4096)
    # Only validated foreground samples enter device geometry buffers. Small
    # per-point observation Jacobians are temporary, never point-point arrays.
    for start in range(0,len(pixels),size):
        ids=pixels[start:start+size];index=backend.tensor(ids,torch.long)
        z=backend.tensor(np.asarray(frame.depth).ravel()[ids]);x=(index%frame.camera.width).double();y=(index//frame.camera.width).double()
        ray=torch.stack([(x-cx)/fx,(y-cy)/fy,torch.ones_like(z)],dim=1)
        world=rigid_transform(ray*z[:,None]-cw[:3,3],rotation,torch.zeros(3,dtype=torch.float64,device=backend.device))
        jac=torch.zeros((len(ids),6,6),dtype=torch.float64,device=backend.device)
        jac[:,:3,0]=rotation[:,0]*z[:,None]/fx;jac[:,:3,1]=rotation[:,1]*z[:,None]/fy
        jac[:,:3,2]=rigid_transform(ray,rotation,torch.zeros(3,dtype=torch.float64,device=backend.device));jac[:,3:,3:]=torch.eye(3,dtype=torch.float64,device=backend.device)
        raw_covariance=covariance(config,z,backend)
        xyz.append(numpy(world));noise.append(numpy(jac@raw_covariance@jac.transpose(1,2)))
    return np.concatenate(xyz),frame.rgb.reshape(-1,3)[pixels],np.concatenate(noise),pixels


def canonicalize(xyz,cov,pose):
    r=pose[:3,:3];p=(xyz-pose[:3,3])@r
    j=np.eye(6);j[:3,:3]=r.T
    return p,j@cov@j.T


def fuse(model,ids,xyz,rgb,cov,raw_ids,backend=None):
    ids=np.asarray(ids,dtype=np.int64);raw_ids=np.asarray(raw_ids,dtype=np.int64)
    if len(ids)!=len(np.unique(ids)):raise ValueError('Fuse one canonical point per camera batch; aggregate repeated factors first')
    if len(np.unique(raw_ids))!=len(raw_ids):raise ValueError('Duplicate raw measurement in one fusion batch')
    owners=model.source_owners(raw_ids)
    if np.any((owners>=0)&(owners!=ids)):raise ValueError('Raw source already belongs to another canonical point')
    use=owners<0
    ids=ids[use];raw_ids=raw_ids[use];xyz=np.asarray(xyz)[use];rgb=np.asarray(rgb)[use];cov=np.asarray(cov)[use]
    if not len(ids):return 0
    old=_inverse(model.covariance[ids],backend);new=_inverse(cov,backend)
    posterior=_inverse(old+new,backend)
    rhs=np.einsum('nij,nj->ni',old,np.c_[model.xyz[ids],model.rgb[ids]])+np.einsum('nij,nj->ni',new,np.c_[xyz,rgb])
    mean=np.einsum('nij,nj->ni',posterior,rhs)
    model.xyz[ids]=mean[:,:3];model.rgb[ids]=mean[:,3:];model.covariance[ids]=posterior
    model.add_sources(ids,raw_ids)
    if hasattr(model,'_registry'):del model._registry
    return len(ids)


def fuse_many(model,ids,xyz,rgb,cov,raw_ids,backend=None):
    """Vectorized conditional fusion of repeated point IDs, raw IDs once only."""
    ids=np.asarray(ids,dtype=np.int64);raw_ids=np.asarray(raw_ids,dtype=np.int64)
    if len(np.unique(raw_ids))!=len(raw_ids):raise ValueError('Duplicate raw measurement in one fusion batch')
    owners=model.source_owners(raw_ids)
    if np.any((owners>=0)&(owners!=ids)):raise ValueError('Raw source already belongs to another canonical point')
    use=owners<0
    ids=ids[use];raw_ids=raw_ids[use];xyz=np.asarray(xyz)[use];rgb=np.asarray(rgb)[use];cov=np.asarray(cov)[use]
    if not len(ids):return np.empty(0,dtype=np.int64)
    unique,group=np.unique(ids,return_inverse=True)
    if backend is None:
        precision=np.linalg.inv(cov);information=np.linalg.inv(model.covariance[unique]);rhs=np.einsum('nij,nj->ni',information,np.c_[model.xyz[unique],model.rgb[unique]])
        np.add.at(information,group,precision);np.add.at(rhs,group,np.einsum('nij,nj->ni',precision,np.c_[xyz,rgb]))
        posterior=np.linalg.inv(information);mean=np.einsum('nij,nj->ni',posterior,rhs)
    else:
        from .backends.torch_cuda import stable_segment_sum
        precision=torch.linalg.inv(backend.tensor(cov));old=torch.linalg.inv(backend.tensor(model.covariance[unique]))
        keys=backend.tensor(group,torch.long)
        information=old+stable_segment_sum(precision,keys,len(unique))
        rhs=torch.einsum('nij,nj->ni',old,backend.tensor(np.c_[model.xyz[unique],model.rgb[unique]]))
        rhs+=stable_segment_sum(torch.einsum('nij,nj->ni',precision,backend.tensor(np.c_[xyz,rgb])),keys,len(unique))
        posterior_tensor=torch.linalg.inv(information);mean=numpy(torch.einsum('nij,nj->ni',posterior_tensor,rhs));posterior=numpy(posterior_tensor)
    model.xyz[unique]=mean[:,:3];model.rgb[unique]=mean[:,3:];model.covariance[unique]=posterior
    model.add_sources(ids,raw_ids)
    if hasattr(model,'_registry'):del model._registry
    return unique


def fuse_frame_zero(dataset,config,backend):
    model=PointMap();raw=ObservationIDs(dataset);records=[]
    state=State(np.eye(4)[None],np.zeros((1,6)))
    for ci in progress(range(len(dataset.cameras)),desc='Fuse frame-0 point observations',unit='camera'):
        frame=dataset.observation(0,ci);data=backproject(frame,config,backend)
        x,color,cov,pixels=data;claimed=np.zeros(len(pixels),bool);associated=0
        if model.count:
            matches,*_=match_rgbd(model,state,frame,config,backend)
            pix=numpy(matches.pixel);point=numpy(matches.point)
            if len(pix):
                location=np.searchsorted(pixels,pix)
                associated=fuse(model,point,x[location],color[location],cov[location],raw.encode(0,ci,pix),backend);claimed[location]=True
        model.append(x[~claimed],color[~claimed],np.zeros((~claimed).sum(),np.int32),cov[~claimed],raw.encode(0,ci,pixels[~claimed]))
        records.append({'camera':ci,'raw_foreground_valid':len(pixels),'fused_existing':associated,'added':int((~claimed).sum()),'points':model.count,'map_bytes':model.nbytes})
    if not model.count:raise ValueError('NO_FRAME_ZERO_GEOMETRY: no valid foreground depth across cameras')
    return model,records


def source_latent_covariance(world,source_pose,measurement_covariance,pose_covariance,measurement_pose_cross=None):
    """Selected augmentation blocks; shared source variable is never copied."""
    from .lie import skew
    r=source_pose[:3,:3].T;j=r@np.c_[-np.eye(3),skew(world)]
    if measurement_pose_cross is None:raise ValueError('Supply raw/source-pose cross covariance; pass an explicit zero block only under a declared independence assumption')
    cross=np.asarray(measurement_pose_cross)
    pp=r@measurement_covariance@r.T+j@pose_covariance@j.T+r@cross@j.T+j@cross.T@r.T
    return pp,j@pose_covariance+r@cross,j


def refuse_original_factors(model,states,dataset,config,backend,indices=None):
    """Rebuild geometry from its original prior and unique registered raw factors.

    Previous coordinate covariance is an initialization, never a second prior.
    Shared pose errors stay in sparse raw-source factors, outside conditional
    precision accumulation with the currently frozen trajectory.
    """
    indices=np.arange(model.count) if indices is None else np.asarray(indices,dtype=np.int64)
    model.source_owners([])  # validates unique raw sources across points
    result=model.copy();n=len(indices);prior=np.linalg.inv(config.map_covariance)
    information=np.broadcast_to(prior,(n,6,6)).copy();rhs=np.broadcast_to(prior@config.map_mean,(n,6)).copy()
    selected=model.provenance.take(indices);raw_ids=selected.raw_ids;count=len(raw_ids)
    local=np.repeat(np.arange(n,dtype=np.int64),selected.lengths)
    raw=ObservationIDs(dataset);times,cameras,pixels=raw.decode(raw_ids)
    for k in progress(np.unique(times),desc='Refuse original point/source factors',unit='time',leave=False):
        for ci in np.unique(cameras[times==k]):
            take=np.flatnonzero((times==k)&(cameras==ci));target=local[take]
            frame=dataset.observation(int(k),int(ci));x,a,c,_=backproject(frame,config,backend,pixels[take])
            labels=model.group[indices[target]]
            for g in np.unique(labels):
                subset=labels==g;x[subset],c[subset]=canonicalize(x[subset],c[subset],states[int(k)].poses[int(g)])
            precision=np.linalg.inv(c)
            np.add.at(information,target,precision);np.add.at(rhs,target,np.einsum('nij,nj->ni',precision,np.c_[x,a]))
    conditional=np.linalg.inv(information);mean=np.einsum('nij,nj->ni',conditional,rhs)
    result.xyz[indices]=mean[:,:3];result.rgb[indices]=mean[:,3:];result.covariance[indices]=conditional
    return result,indices,{'raw_factors':count,'points':n,'prior':'original documented canonical Gaussian, once','covariance':'conditional on frozen group poses and source assignments; source pose correlations remain shared latent factors'}
