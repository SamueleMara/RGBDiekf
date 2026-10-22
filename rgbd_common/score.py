"""One normalized fixed-unit point/assignment score for both complete pipelines."""
import numpy as np
import torch
from .matching_cuda import match_rgbd
from .observation_model import unassigned_log_probability
from .lie import exp,log,inverse
from .backends.torch_cuda import numpy
from .progress import progress


def gaussian_nll(residual,covariance):
    residual=np.asarray(residual)
    if residual.ndim==2:
        return float(.5*(np.sum(residual*np.linalg.solve(covariance,residual.T).T)+len(residual)*(np.linalg.slogdet(covariance)[1]+residual.shape[1]*np.log(2*np.pi))))
    return float(.5*(residual@np.linalg.solve(covariance,residual)+np.linalg.slogdet(covariance)[1]+len(residual)*np.log(2*np.pi)))


def frame_score(model,state,dataset,k,config,backend,ownership=None):
    value=0.;records=[];matches=[]
    for ci in range(len(dataset.cameras)):
        frame=dataset.observation(k,ci);match,stats,_,_=match_rgbd(model,state,frame,config,backend)
        if hasattr(dataset,'background_total'):background=dataset.background_total(k,ci,config,backend)
        else:
            bg=unassigned_log_probability(frame,config,backend)+np.log1p(-config.match_prior)
            mask=backend.tensor(frame.valid.ravel(),torch.bool);background=float(bg[mask].sum())
        cost=-background-float((match.assignment_score-match.unmatched_score).sum())
        if not stats['mask_feasible']:cost=float('inf')
        if ownership is not None:ownership.write(k,ci,numpy(match.pixel),model.point_id[numpy(match.point)])
        value+=cost;matches.append(match)
        records.append({'frame':k,'camera':ci,'energy':cost,'foreground':int(frame.mask.sum()),'foreground_valid':int(frame.valid.sum()),'unsolved':int(frame.valid.sum())-len(match),**stats})
    return value,matches,records


def priors(model,states,times,config):
    groups=len(states[0].poses);motion=0.
    for g in range(groups):
        motion+=gaussian_nll(states[0].twists[g]-config.twist_mean,config.twist_covariance)
    for k in range(len(states)-1):
        dt=times[k+1]-times[k];q=config.q(dt,groups)
        residual=np.array([np.r_[log(inverse(exp(dt*v)@a)@b),w-v] for a,b,v,w in zip(states[k].poses,states[k+1].poses,states[k].twists,states[k+1].twists)]).ravel()
        motion+=gaussian_nll(residual,q)
    error=np.c_[model.xyz,model.rgb]-config.map_mean
    geometry=float(.5*np.sum(error*np.linalg.solve(config.map_covariance,error.T).T)+.5*model.count*(np.linalg.slogdet(config.map_covariance)[1]+6*np.log(2*np.pi)))
    from .model_contract import require_model_selection
    require_model_selection(config)
    point_structure,structure=config.structure_prior(model,states,times)
    geometry+=point_structure
    return {'motion_prior':motion,'geometry_prior':geometry,'group_prior':float(structure)}


def score_full_sequence(model,states,dataset,config,backend,ownership=None,description='Full common point score'):
    value=0.;records=[]
    for k in progress(range(len(dataset.times)),desc=description,unit='frame'):
        cost,_,rows=frame_score(model,states[k],dataset,k,config,backend,ownership)
        value+=cost;records.extend(rows)
    prior=priors(model,states,dataset.times,config)
    return {'energy':float(value+sum(prior.values())),'observation_energy':value,**prior,'frames':records,
            'observation_count':sum(row['foreground_valid'] for row in records),'solved_points':model.count,'groups':len(states[0].poses)}


def transactional_accept(old_score,new_score,commit,log_proposal):
    accepted=bool(np.isfinite(new_score['energy']) and new_score['energy']<=old_score['energy'])
    log_proposal({'old_energy':old_score['energy'],'new_energy':new_score['energy'],'accepted':accepted})
    if accepted:commit()
    return accepted
