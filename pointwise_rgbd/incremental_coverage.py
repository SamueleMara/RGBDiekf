"""Monotone frozen-surface coverage, using only newly inserted trajectories.

The initial candidate family is observed->SOLVED spatial kNN. Thereafter each
new trajectory is tested against unresolved samples; valid old witnesses are
retained, even if a newly inserted incompatible point would displace a witness
from a recomputed top-k list. This is an exact union over that declared candidate
family, not a claim to exhaustively search every old point.

The disk cache holds measured foreground observations (UNSOLVED lookup data),
not per-pixel latent states or a new eager all-frame canonical catalogue.
"""
import json,time
from types import SimpleNamespace
import numpy as np
from scipy.spatial import cKDTree
from tqdm.auto import tqdm
from .coverage import SurfaceIndex
from .observations import Samples,samples
from .projection import backproject


def cache_observations(index,k,ci,frame,config,record,raw=None):
    """One-time immutable lookup data, initialized alongside base coverage."""
    raw=samples(frame,config) if raw is None else raw
    if not np.array_equal(raw.pixel,record['pixel']):raise ValueError('Coverage/cache raw identity mismatch')
    use=~record['explained'];cov=raw.covariance[use]
    maximum=float(np.linalg.eigvalsh(cov)[:,-1].max()) if len(cov) else 0.
    index.store.write(f'_support:{k}:{ci}',pixel=raw.pixel,depth=raw.z[:,2],mask_count=np.array(frame.mask.sum()))
    index.store.write(f'_unresolved:{k}:{ci}',pixel=raw.pixel[use],z=raw.z[use],xyz=raw.xyz[use],rgb=raw.rgb[use],max_cov_eigenvalue=np.array(maximum))


def euclidean_gate_radius(p,observation_max_eigenvalue,chi_square):
    """Necessary Euclidean bound for e' [2(P+Sigma)]^-1 e <= chi2.

    Weyl's inequality bounds lambda_max(P+Sigma); no metric tuning constant.
    nextafter rounds the search radius outward, not an extra physical tolerance.
    """
    eigen=np.linalg.eigvalsh(p)
    if not np.isfinite(eigen).all() or eigen[0]<0:return None
    return np.nextafter(np.sqrt(chi_square*2*(eigen[-1]+observation_max_eigenvalue)),np.inf)


def update_insertions(dataset,ownership,config,backend,debug=None):
    begin=time.monotonic();index=ownership.coverage;horizon=index.horizon()
    version=ownership.meta('model_version',0)
    if horizon is None or int(horizon['version'])<0:raise ValueError('Incremental coverage requires a completed base')
    last=int(horizon['last_point_id'])
    pending=list(ownership.db.execute('SELECT id,path,data FROM points WHERE id>? ORDER BY id',(last,)))
    if not pending:raise ValueError('Coverage version changed without a registered new trajectory')
    totals={'model_version':version,'new_trajectories':len(pending),'old_trajectory_frame_reads':0,'raw_image_loads':0,'candidate_pairs_tested':0,'new_appearance_coverage':0,'new_geometry_coverage':0,'views_changed':0,'views_visited':0}
    # Only the delta tracks are opened. No TrackStore.frame or old NPZ is used.
    tracks=[]
    for point,path,metadata in pending:
        with np.load(path) as data:tracks.append((point,data['state'],data['covariance'],np.array(json.loads(metadata)['color'])))
    for k in tqdm(range(len(dataset.times)-1,-1,-1),desc='Update coverage from NEW trajectories only',unit='frame',leave=False):
        surfaces=[SurfaceIndex(s[k:k+1],p[k:k+1],rgb[None],np.array([point])) for point,s,p,rgb in tracks]
        for ci,camera in enumerate(dataset.cameras):
            key=f'{k}:{ci}';record=index.store.read(key);totals['views_visited']+=1
            if int(record['model_version'])==version:continue # Atomic per-view resume.
            if record['explained'].all():continue
            support=index.store.read(f'_support:{k}:{ci}')
            valid=np.zeros(camera.height*camera.width,bool);valid[support['pixel']]=True
            depth=np.full(valid.size,np.nan);depth[support['pixel']]=support['depth']
            view=SimpleNamespace(camera=camera,k=k,ci=ci,depth=depth.reshape(camera.height,camera.width),valid=valid.reshape(camera.height,camera.width),mask=valid.reshape(camera.height,camera.width))
            base=index.store.read(f'_unresolved:{k}:{ci}')
            tree=None;changed=False
            for surface in surfaces:
                # Visibility rejection before any observed-point search.
                if surface.projection(view,config,backend)[2][0]!=0:continue
                radius=euclidean_gate_radius(surface.p[0],float(base['max_cov_eigenvalue']),config.gate(3))
                if radius is None:continue
                if tree is None:tree=cKDTree(base['xyz'])
                rows=np.asarray(tree.query_ball_point(surface.state[0,:3],radius),dtype=np.int64)
                if not len(rows):continue
                positions=np.searchsorted(record['pixel'],base['pixel'][rows])
                use=~record['explained'][positions];rows=rows[use];positions=positions[use]
                if not len(rows):continue
                z=base['z'][rows];r=config.measurement_covariance(z[:,2]);xyz,cov=backproject(z,camera,r)
                raw=Samples(view,base['pixel'][rows],z,xyz,cov,r,base['rgb'][rows],None)
                result=surface.match(view,config,backend,raw)
                totals['candidate_pairs_tested']+=len(rows)
                for mask,witness,count in [('geometry_explained','geometry_point_id','new_geometry_coverage'),('explained','point_id','new_appearance_coverage')]:
                    accepted=getattr(result,mask)&~record[mask][positions]
                    ids=positions[accepted]
                    if len(ids):
                        record[mask][ids]=True;record[witness][ids]=getattr(result,witness)[accepted];changed=True;totals[count]+=len(ids)
            if changed:
                record['model_version']=np.array(version);index.store.write(key,**record);totals['views_changed']+=1
    index.complete(version,max(point for point,_,_ in pending))
    totals['seconds']=time.monotonic()-begin
    if debug:
        from .store import json_write
        json_write(debug.root/f'discovery/coverage_updates/version_{version:06d}.json',totals)
    return totals
