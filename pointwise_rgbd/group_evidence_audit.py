"""Read-only sensor audit of cached group histories, with no estimator updates.

Accepted factors are selection-conditioned training evidence. Replayed rejected
candidates are explicitly retrospective, not invented historical log entries.
No physical labels are inferred here.
"""
from io import BytesIO
from pathlib import Path
import json,sqlite3
import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from .surface_supernodes import Partition
from .observations import samples
from .projection import backproject


REASONS=('ACCEPTED','OUTSIDE_CAMERA','INVALID_MASK_OR_DEPTH','Z_OCCLUDED',
         'GEOMETRIC_GATE','RAW_ID_OWNED','NUMERIC_TIE','NO_LOCAL_CANDIDATE')


def read_factors(root,k):
    """Never CREATE tables or journal files in a preserved experiment."""
    with sqlite3.connect((Path(root)/'actual_measurement_factors.sqlite').resolve().as_uri()+'?mode=ro',uri=True) as db:
        row=db.execute('SELECT value FROM arrays WHERE key=?',(str(k),)).fetchone()
    if row is None:return None
    with np.load(BytesIO(row[0]),allow_pickle=False) as values:return {key:values[key] for key in values.files}


def load_cached(root):
    root=Path(root)
    with np.load(root/'partition.npz',allow_pickle=False) as values:
        p=Partition(**{key:values[key].copy() for key in Partition.__dataclass_fields__})
    arrays={key:np.load(root/(key+'.npy'),mmap_mode='r') for key in
            ('group_poses','covariance','xyz','point_observed','group_measurements','pose_rank')}
    if (root/'twists.npy').exists():arrays['twists']=np.load(root/'twists.npy',mmap_mode='r')
    return p,arrays


def projection_numpy(xyz,camera):
    r=camera.world_to_camera[:3,:3];q=xyz@r.T+camera.world_to_camera[:3,3]
    fx,fy,cx,cy=camera.intrinsics
    with np.errstate(divide='ignore',invalid='ignore'):
        h=np.c_[fx*q[:,0]/q[:,2]+cx,fy*q[:,1]/q[:,2]+cy,q[:,2]]
        j=np.zeros((len(q),3,3));j[:,0,0]=fx/q[:,2];j[:,0,2]=-fx*q[:,0]/q[:,2]**2
        j[:,1,1]=fy/q[:,2];j[:,1,2]=-fy*q[:,1]/q[:,2]**2;j[:,2,2]=1
    return h,j@r


def skew_many(x):
    g=np.zeros((len(x),3,3));g[:,0,1]=-x[:,2];g[:,0,2]=x[:,1];g[:,1,0]=x[:,2]
    g[:,1,2]=-x[:,0];g[:,2,0]=-x[:,1];g[:,2,1]=x[:,0];return g


def position_jacobian(x):
    return np.concatenate((np.broadcast_to(np.eye(3),(len(x),3,3)),-skew_many(x)),axis=2)


def quantiles(x):
    x=np.asarray(x);x=x[np.isfinite(x)]
    return dict(zip(('p50','p95','p99','max'),map(float,np.quantile(x,[.5,.95,.99,1])))) if len(x) else None


def replay_candidates(xyz,position_covariance,colors,raw,config,exclusive=True):
    """Same finite kNN family/gates/cost as baseline; retain every candidate.

    Classification is at the supplied pose. RGB has a Gaussian ranking term,
    *no* independent RGB rejection gate in the historical implementation.
    """
    from .association import exclusive_assignment
    n=len(xyz);camera=raw.frame.camera;h,j=projection_numpy(xyz,camera)
    reason=np.full(n,REASONS.index('OUTSIDE_CAMERA'),np.uint8)
    finite=np.isfinite(h).all(1);uv=np.zeros((n,2),np.int64);uv[finite]=np.floor(h[finite,:2]+.5).astype(np.int64)
    inside=finite&(h[:,2]>0)&(uv[:,0]>=0)&(uv[:,0]<camera.width)&(uv[:,1]>=0)&(uv[:,1]<camera.height)
    ids=np.flatnonzero(inside);pixel=uv[ids,1]*camera.width+uv[ids,0]
    valid=raw.frame.valid.ravel()[pixel];depth=np.where(valid,np.asarray(raw.frame.depth).ravel()[pixel],0)
    variance=np.einsum('ni,nij,nj->n',j[ids,2],position_covariance[ids],j[ids,2])+config.measurement_covariance(depth)[:,2,2]
    hidden=valid&(depth<h[ids,2])&((depth-h[ids,2])**2>config.gate(1)*variance)
    reason[ids]=REASONS.index('NO_LOCAL_CANDIDATE');reason[ids[~valid]]=REASONS.index('INVALID_MASK_OR_DEPTH')
    reason[ids[hidden]]=REASONS.index('Z_OCCLUDED');ids=ids[~hidden]
    empty={'point':np.empty(0,np.int64),'pixel':np.empty(0,np.int64),'z':np.empty((0,3)),
           'S':np.empty((0,3,3)),'residual':np.empty((0,3)),'d2':np.empty(0),'ray_d2':np.empty(0),
           'rgb_d2':np.empty(0),'rgb_cosine':np.empty(0),'rgb_distance':np.empty(0),
           'cost':np.empty(0),'metric_distance_m':np.empty(0),'reason':np.empty(0,np.uint8)}
    if not len(ids) or raw.tree is None:return reason,empty
    distance,near=raw.tree.query(xyz[ids],k=min(config.neighbors,len(raw.pixel)),workers=1)
    if near.ndim==1:near=near[:,None];distance=distance[:,None]
    point=np.repeat(ids,near.shape[1]);near=near.ravel();pixel=raw.pixel[near]
    e=raw.z[near]-h[point];s=j[point]@position_covariance[point]@j[point].transpose(0,2,1)+raw.R[near]
    d2=np.einsum('ni,ni->n',e,np.linalg.solve(s,e[:,:,None])[:,:,0])
    ray=np.einsum('ni,ni->n',e[:,:2],np.linalg.solve(s[:,:2,:2],e[:,:2,None])[:,:,0])
    rgb=raw.rgb[near]-colors[point];affinity=np.einsum('ni,ij,nj->n',rgb,np.linalg.inv(2*config.color_covariance),rgb)
    denom=np.linalg.norm(raw.rgb[near],axis=1)*np.linalg.norm(colors[point],axis=1)
    cosine=np.divide(np.einsum('ni,ni->n',raw.rgb[near],colors[point]),denom,out=np.full(len(point),np.nan),where=denom>0)
    cost=.5*(d2+np.linalg.slogdet(s)[1]+affinity)
    good=(d2<=config.gate(3))&(ray<=config.gate(2));cost[~good]=np.inf
    order=np.lexsort((pixel,cost,point));q=point[order];first=np.r_[True,q[1:]!=q[:-1]];rows=np.flatnonzero(first)
    second=rows+1;has=second<len(order);has[has]&=q[second[has]]==q[rows[has]]
    a=order[rows[has]];b=order[second[has]];finite_pair=np.isfinite(cost[a])&np.isfinite(cost[b]);tie=np.zeros(len(a),bool)
    tie[finite_pair]=abs(cost[a[finite_pair]]-cost[b[finite_pair]])<=8*np.finfo(float).eps*np.maximum(1,abs(cost[a[finite_pair]]))
    tied=np.zeros(n,bool);tied[point[a[tie]]]=True
    eligible=np.flatnonzero(good&~tied[point])
    if exclusive:selected=eligible[exclusive_assignment(point[eligible],pixel[eligible],cost[eligible],np.arange(n))]
    else:
        ranking=np.lexsort((pixel[eligible],cost[eligible],point[eligible]));_,first=np.unique(point[eligible[ranking]],return_index=True);selected=eligible[ranking[first]]
    candidate_reason=np.full(len(point),REASONS.index('GEOMETRIC_GATE'),np.uint8)
    candidate_reason[good]=REASONS.index('RAW_ID_OWNED');candidate_reason[tied[point]]=REASONS.index('NUMERIC_TIE');candidate_reason[selected]=0
    reason[np.unique(point)]=REASONS.index('GEOMETRIC_GATE');reason[np.unique(point[good])]=REASONS.index('RAW_ID_OWNED')
    reason[tied]=REASONS.index('NUMERIC_TIE');reason[point[selected]]=0
    return reason,dict(point=point,pixel=pixel,z=raw.z[near],S=s,residual=e,d2=d2,ray_d2=ray,
        rgb_d2=affinity,rgb_cosine=cosine,rgb_distance=np.linalg.norm(rgb,axis=1),cost=cost,
        metric_distance_m=distance.ravel(),reason=candidate_reason)


def group_information(partition,poses,covariance,factors,cameras):
    """Dimensionless information uses source XYZ noise and patch radius.

    Translation scale sigma metres, angle scale sigma/L radians. Angular Schur
    complement removes translations. Unobservable eigenmodes are reported, not
    filled by a small diagonal. Conditional calibration/canonical geometry.
    """
    s=partition.count;n=len(partition.xyz);member=partition.membership
    centroids=np.einsum('sij,sj->si',poses[:,:3,:3],partition.centroids)+poses[:,:3,3]
    information=np.zeros((s,6,6));points=np.zeros(n,bool);camera_support=np.zeros((s,len(cameras)),bool)
    if factors is not None:
        ids=factors['point_id'];points[ids]=True
        for ci,camera in enumerate(cameras):
            select=factors['camera']==ci;idx=ids[select];g=member[idx]
            xyz=np.einsum('nij,nj->ni',poses[g,:3,:3],partition.xyz[idx])+poses[g,:3,3]
            _,j=projection_numpy(xyz,camera);h=j@position_jacobian(xyz)
            block=h.transpose(0,2,1)@np.linalg.solve(factors['R'][select],h)
            np.add.at(information,g,block);camera_support[g,ci]=True
    # delta t_center = delta rho - [center]_x delta theta.
    world_from_center=np.repeat(np.eye(6)[None],s,axis=0);world_from_center[:,:3,3:]=skew_many(centroids)
    centered=world_from_center.transpose(0,2,1)@information@world_from_center
    sigma=np.sqrt(np.bincount(member,weights=np.trace(partition.source_covariance,axis1=1,axis2=2)/3,minlength=s)/np.bincount(member,minlength=s))
    spread=partition.radius
    scales=np.c_[np.repeat(sigma[:,None],3,axis=1),np.repeat(np.divide(sigma,spread,out=np.zeros(s),where=spread>0)[:,None],3,axis=1)]
    normalized=centered*scales[:,:,None]*scales[:,None,:]
    spectra=np.linalg.eigvalsh(normalized)
    schur=centered[:,3:,3:]-centered[:,3:,:3]@np.linalg.pinv(centered[:,:3,:3],hermitian=True)@centered[:,:3,3:]
    angular_spectra=np.linalg.eigvalsh((schur+schur.transpose(0,2,1))/2)
    tol=6*np.finfo(float).eps*np.maximum(spectra[:,-1],0);rank=(spectra>tol[:,None]).sum(1)
    condition=np.divide(spectra[:,-1],spectra[:,0],out=np.full(s,np.inf),where=spectra[:,0]>tol)
    center_map=np.linalg.inv(world_from_center);pc=center_map@covariance[:,:6,:6]@center_map.transpose(0,2,1)
    # Gain refers to posterior relative to raw measurement information only;
    # no old mixed-unit eigenvalue ratios or claims about prior independence.
    gain=np.linalg.slogdet(np.eye(6)[None]+covariance[:,:6,:6]@information)[1]/2
    return dict(centroid=centroids,radius_m=spread,group_size=np.bincount(member,minlength=s),
        distinct_points=np.bincount(member,weights=points,minlength=s).astype(int),distinct_cameras=camera_support.sum(1),
        normalized_information_eigenvalues=spectra,normalization_translation_m=sigma,
        normalization_angle_rad=scales[:,3],normalized_condition=condition,
        angular_schur_information=angular_spectra,center_translation_std_m=np.sqrt(np.maximum(0,np.diagonal(pc[:,:3,:3],axis1=1,axis2=2))),
        angular_std_rad=np.sqrt(np.maximum(0,np.diagonal(covariance[:,3:6,3:6],axis1=1,axis2=2))),
        conditional_measurement_gain_nats=gain,
        status=np.where(rank<6,'POSE_ROTATION_UNOBSERVABLE','NUMERICAL_FULL_RANK_NOT_PRECISION_CERTIFIED'),
        information_world=information)


def connected_components_of_partition(partition):
    n=len(partition.xyz);edge=partition.edges
    graph=coo_matrix((np.ones(2*len(edge)),(np.r_[edge[:,0],edge[:,1]],np.r_[edge[:,1],edge[:,0]])),shape=(n,n)).tocsr()
    return connected_components(graph,directed=False)[1]
