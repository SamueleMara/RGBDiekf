import numpy as np
import torch
from tqdm.auto import tqdm
from .observations import initialize_seed
from .process6 import predict
from .association import associate
from .filter_iekf import correct


def track_one_point(point_id,source,dataset,config,backend,ownership):
    state,p,color=initialize_seed(source,dataset,config);anchor=state.copy();anchor_p=p.copy()
    frames=len(dataset.times)
    track={'state':np.empty((frames,6)),'covariance':np.empty((frames,6,6)),
        'observed':np.zeros(frames,bool),'nis':np.full(frames,np.nan),'counts':np.zeros(frames,np.uint8),
        'associations':[],'matching_stats':[],'filter_reports':[],'backward_frames':[],'forward_frames':[],'anchor_uses':1,'numerical_failure':False}
    track['state'][source.frame]=state;track['covariance'][source.frame]=p;track['observed'][source.frame]=True;track['counts'][source.frame]=1
    claims=[(source.frame,source.camera,np.array([source.pixel]),np.array([point_id]))]
    track['associations'].append({'frame':source.frame,'camera':source.camera,'pixel':source.pixel,'reason':'ANCHOR_INITIALIZED_ONCE','point_id':point_id})
    for name,indices,reverse in [('backward',range(source.frame-1,-1,-1),True),('forward',range(source.frame+1,frames),False)]:
        # Both branches restart from independent copies of the anchor posterior.
        current=backend.tensor(anchor[None]).clone();cov=backend.tensor(anchor_p[None]).clone();previous=source.frame
        for k in tqdm(indices,desc=f'Point {point_id} {name}',unit='frame',leave=False):
            dt=abs(dataset.times[k]-dataset.times[previous]);prior,pminus=predict(current,cov,dt,config,reverse)
            matches=[]
            for ci in range(len(dataset.cameras)):
                frame=dataset.observation(k,ci)
                match,stats=associate(prior.cpu().numpy(),pminus.cpu().numpy(),color[None],np.array([point_id]),frame,config,backend,ownership.get(k,ci))
                matches.append(match);track['matching_stats'].append(stats)
                track['associations'].extend(stats['candidates'])
                if len(match['pixel']):
                    pixel=int(match['pixel'][0]);claims.append((k,ci,np.array([pixel]),np.array([point_id])))
                    from .state6 import ObservationRef
                    ownership.reserve(ObservationRef(k,ci,pixel),point_id)
                else:
                    track['associations'].append({'frame':k,'camera':ci,'pixel':None,'point_id':point_id,'reason':'OCCLUDED' if stats['occluded'] else ('AMBIGUOUS' if stats['ambiguous'] else 'NO_COMPATIBLE_UNOWNED_FOREGROUND')})
            current,cov,report=correct(prior,pminus,matches,config)
            if not bool(torch.isfinite(current).all()&torch.isfinite(cov).all()):raise ArithmeticError('Nonfinite new point trajectory')
            track['state'][k]=current.cpu().numpy()[0];track['covariance'][k]=cov.cpu().numpy()[0]
            count=int(report['measurement_count'][0]);track['counts'][k]=count;track['observed'][k]=count>0
            track['filter_reports'].append({'frame':k,'direction':name,'raw_measurements':count,'iterations':int(report['iterations'][0]),'energy_before':float(report['before'][0]),'energy_after':float(report['after'][0]),'line_search_stops':int(report['line_search_stops'][0]),'iteration_limit_reached':bool(report['iteration_limit_reached'][0]),'status':'REAL_MEASUREMENT' if count else 'PREDICTION_ONLY'})
            if count:track['nis'][k]=float(report['nis'][0])
            track[name+'_frames'].append(k);previous=k
    assert track['backward_frames']==list(range(source.frame-1,-1,-1))
    assert track['forward_frames']==list(range(source.frame+1,frames))
    refs=[(k,c,int(p)) for k,c,pixels,_ in claims for p in pixels]
    if len(set(refs))!=len(refs):raise AssertionError('An anchor/raw observation was counted twice')
    return track,claims,color
