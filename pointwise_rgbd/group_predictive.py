"""Same-identity held-out UVZ scoring, including shared group pose uncertainty.

The sum across frames is a sum of marginal proper predictive scores. It is NOT
a joint temporal log likelihood: cross-time covariance is not discarded under
an independence claim. Archived correspondence selection remains conditioned.
"""
import numpy as np
import torch
from rgbd_common.backends.torch_cuda import stable_segment_sum
from .projection import project
from .group_pose import transport,point_jacobian


def heldout_uvz_score(poses,covariance,canonical,membership,matches,backend):
    groups=len(poses);p=covariance[:,:6,:6];precision=torch.linalg.inv(p)
    a=precision.clone();b=poses.new_zeros((groups,6));square=poses.new_zeros(groups);logdet=poses.new_zeros(groups)
    error_position=poses.new_zeros(groups);counts=torch.zeros(groups,device=poses.device,dtype=torch.long);residuals=[]
    xyz=transport(canonical,membership,poses)
    for match in matches:
        ids=match['point'];g=membership[ids]
        if not len(ids):continue
        h,j=project(xyz[ids],match['camera']);e=match['z']-h;H=j[:,:,:3]@point_jacobian(xyz[ids]);R=match['R']
        w=torch.linalg.solve(R,H);rhs=torch.linalg.solve(R,e[:,:,None])[:,:,0]
        a+=stable_segment_sum(H.transpose(1,2)@w,g,groups)
        b+=stable_segment_sum((H.transpose(1,2)@rhs[:,:,None])[:,:,0],g,groups)
        square+=stable_segment_sum((e*rhs).sum(1),g,groups);logdet+=stable_segment_sum(torch.linalg.slogdet(R).logabsdet,g,groups)
        error_position+=stable_segment_sum(e[:,2]**2,g,groups);counts+=torch.bincount(g,minlength=groups)
        residuals.append(e.cpu().numpy())
    q=square-(b*torch.linalg.solve(a,b[:,:,None])[:,:,0]).sum(1)
    determinant=logdet+torch.linalg.slogdet(p).logabsdet+torch.linalg.slogdet(a).logabsdet
    nll=.5*(q+determinant+3*counts.to(poses.dtype)*np.log(2*np.pi));empty=counts==0;nll[empty]=0;q[empty]=0
    return {'nll_per_group':nll.cpu().numpy(),'chi2_per_group':q.cpu().numpy(),'factor_count_per_group':counts.cpu().numpy(),
        'depth_squared_error_per_group_m2':error_position.cpu().numpy(),'raw_residual_uvz':np.concatenate(residuals) if residuals else np.empty((0,3)),
        'score_semantics':'joint same-frame shared-pose Gaussian; sum across frames is a marginal proper score, not a joint temporal likelihood; selected raw correspondence identities conditioned'}


def matches_from_factors(factors,partition,cameras,backend,heldout):
    """Linked canonical provenance units never cross fit/evaluation folds."""
    select=(partition.duplicate_unit[factors['point_id']]%2==0)==heldout;matches=[]
    for ci,camera in enumerate(cameras):
        use=select&(factors['camera']==ci)
        matches.append({'camera':camera,'camera_index':ci,'point':backend.tensor(factors['point_id'][use],torch.long),
            'pixel':factors['pixel'][use],'z':backend.tensor(factors['z'][use]),'R':backend.tensor(factors['R'][use])})
    return matches
