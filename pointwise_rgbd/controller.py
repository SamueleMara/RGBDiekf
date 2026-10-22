import json,time,resource
from pathlib import Path
import numpy as np
import torch
from tqdm.auto import tqdm
from .store import Ownership,TrackStore,json_write
from .observations import initialize_all_frame_zero
from .process6 import predict
from .association import associate
from .filter_iekf import correct
from .discovery import track_one_point
from .coverage import SurfaceIndex
from .coverage_store import FrozenMeasurementConflict


def initial_forward(dataset,config,backend,root,ownership,debug,resume=False):
    root=Path(root);directory=root/'initial_forward/trajectory_store'
    if resume and (directory/'initial_points.npz').exists():
        store=TrackStore(directory,dataset);start_frame=ownership.meta('initial_next_frame',0)
        if start_frame==0:raise ValueError('Initial checkpoint has no complete anchor frame; preserve files and restart in a fresh output')
        store.validate_completed_inputs(start_frame)
        state=np.array(store.arrays['state'][start_frame-1]);p=np.array(store.arrays['covariance'][start_frame-1])
        ownership.rollback_initial_after(start_frame-1)
    else:
        state,p,color,sources=initialize_all_frame_zero(dataset,config)
        store=TrackStore(directory,dataset,len(state),(color,sources));start_frame=1
        digest=store.write_frame(0,state,p,np.ones(len(state),bool),np.full(len(state),np.nan),np.ones(len(state),np.uint8))
        claims=[]
        for ci in range(len(dataset.cameras)):
            ids=np.flatnonzero(sources[:,1]==ci);claims.append((0,ci,sources[ids,2],ids))
        store.write_measurements(0,[{'point':ids,'pixel':pixels,'camera_index':ci} for _,ci,pixels,ids in claims])
        ownership.claim(claims,kind=3,checkpoint={'initial_next_frame':1,'initial_invocations':1,'initial_frame_digests':[digest],'next_point_id':len(state)})
        debug.initial_anchor(store)
    if ownership.meta('initial_complete',False):
        store.freeze();debug.initial_complete(store,ownership);return store
    n=store.n0
    for k in tqdm(range(start_frame,len(dataset.times)),desc='Pointwise initial forward (all points)',unit='frame'):
        begin=time.monotonic();pred=np.empty_like(state);pp=np.empty_like(p)
        size=min(n,backend.chunk_size(6*6*8*30))
        for lo in range(0,n,size):
            hi=min(n,lo+size);s,cov=predict(backend.tensor(state[lo:hi]),backend.tensor(p[lo:hi]),dataset.times[k]-dataset.times[k-1],config)
            pred[lo:hi]=s.cpu().numpy();pp[lo:hi]=cov.cpu().numpy()
        matches=[];statistics=[]
        for ci in range(len(dataset.cameras)):
            match,stats=associate(pred,pp,store.colors,np.arange(n),dataset.observation(k,ci),config,backend,ownership.tracking_owners(k,ci))
            matches.append(match);statistics.append(stats)
        nis=np.full(n,np.nan);counts=np.zeros(n,np.uint8);filter_summary={'min_eigenvalue':float('inf'),'max_condition':0.,'line_search_stops':0,'iteration_limit_reached':0,'iterations_max':0,'accepted_energy_decreases':0,'psd_repairs':0}
        filter_start=time.monotonic()
        for lo in range(0,n,size):
            hi=min(n,lo+size);local=[]
            for m in matches:
                use=(m['point']>=lo)&(m['point']<hi)
                local.append({'camera':m['camera'],'point':m['point'][use]-lo,'z':m['z'][use],'R':m['R'][use]})
            s,cov,report=correct(backend.tensor(pred[lo:hi]),backend.tensor(pp[lo:hi]),local,config)
            state[lo:hi]=s.cpu().numpy();p[lo:hi]=cov.cpu().numpy();counts[lo:hi]=report['measurement_count'].cpu().numpy()
            observed=counts[lo:hi]>0;nis[lo:hi][observed]=report['nis'].cpu().numpy()[observed]
            # Diagnostic spectra use the already transferred FP64 blocks; the
            # real-size CUDA batched eigensolver failed in the smoke run.
            eigen=np.linalg.eigvalsh(p[lo:hi])
            filter_summary['min_eigenvalue']=min(filter_summary['min_eigenvalue'],float(eigen[:,0].min()))
            filter_summary['max_condition']=max(filter_summary['max_condition'],float((eigen[:,-1]/eigen[:,0]).max()))
            filter_summary['line_search_stops']+=int(report['line_search_stops'].sum());filter_summary['iterations_max']=max(filter_summary['iterations_max'],int(report['iterations'].max()))
            filter_summary['accepted_energy_decreases']+=int((report['after']<report['before']).sum())
            filter_summary['iteration_limit_reached']+=int(report['iteration_limit_reached'].sum())
        backend.synchronize();filter_seconds=time.monotonic()-filter_start
        digest=store.write_frame(k,state,p,counts>0,nis,counts)
        store.write_measurements(k,matches)
        digests=ownership.meta('initial_frame_digests',[]);digests.append(digest)
        claims=[(k,m['camera_index'],m['pixel'],m['point'].cpu().numpy()) for m in matches]
        ownership.claim(claims,checkpoint={'initial_next_frame':k+1,'initial_frame_digests':digests})
        debug.initial_frame(k,statistics,nis,counts,p,filter_summary,filter_seconds,time.monotonic()-begin)
    store.freeze()
    with ownership.db:ownership.set_meta('initial_complete',True)
    debug.initial_complete(store,ownership)
    return store


def match_all_frozen(dataset,store,ownership,config,backend,debug=None):
    """Observed -> SOLVED coverage; no owner claims and NEVER a filter update."""
    version=ownership.meta('model_version',0)
    if ownership.coverage.completed(version):return
    horizon=ownership.coverage.horizon()
    if horizon is not None and int(horizon['version'])>=0:
        from .incremental_coverage import update_insertions
        return update_insertions(dataset,ownership,config,backend,debug)
    from .incremental_coverage import cache_observations
    for k in tqdm(range(len(dataset.times)-1,-1,-1),desc='Surface coverage of frozen trajectories',unit='frame',leave=False):
        state,p,color,point_ids,_=store.frame(k)
        surface=SurfaceIndex(state,p,color,point_ids)
        for ci in range(len(dataset.cameras)):
            frame=dataset.observation(k,ci)
            result=surface.match(frame,config,backend)
            ownership.coverage.write(k,ci,version,result)
            cache_observations(ownership.coverage,k,ci,frame,config,ownership.coverage.read(k,ci,version))
            if debug:debug.coverage_row(version,k,ci,result,ownership.tracking_owners(k,ci))
    last=ownership.db.execute('SELECT MAX(id) FROM points').fetchone()[0]
    ownership.coverage.complete(version,-1 if last is None else last)


def run_exact_pointwise_discovery(dataset,config,backend,directory,max_events=None,resume=False,cached_store=None,coverage_audit=None):
    from .diagnostics import Diagnostics
    config.gate();root=Path(directory);root.mkdir(parents=True,exist_ok=True)
    debug=Diagnostics(root,dataset,config,backend);ownership=Ownership(dataset,root/'raw_observations/ownership_index.sqlite')
    if resume:ownership.release_reservations()
    start=time.monotonic()
    if cached_store is not None:
        from .cached_discovery import prepare_frozen_history,import_coverage_audit
        store=prepare_frozen_history(cached_store,ownership,dataset,debug)
        if coverage_audit is not None:import_coverage_audit(store,ownership,dataset,coverage_audit,debug)
    else:store=initial_forward(dataset,config,backend,root,ownership,debug,resume)
    seen={r['point_id'] for r in store.added}
    for point,path,data in ownership.db.execute('SELECT id,path,data FROM points ORDER BY id'):
        if point in seen:continue
        import hashlib
        record=json.loads(data)
        if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=record['sha256']:raise ValueError('Registered frozen track SHA256 mismatch')
        store.append(path,record)
        event=root/f'discovery/events/{point:012d}'
        if not (event/'local_3d.html').exists():
            with np.load(path) as saved:
                track={key:saved[key] for key in ('state','covariance','observed','nis','counts')}
                track.update(json.loads(str(saved['diagnostics_json'])))
            from .state6 import ObservationRef
            debug.event_complete(point,ObservationRef(*record['source']),ownership.meta('model_version',0),track,store,ownership)
    completed=0;failures=0;status='COMPLETE_FINITE_DISCOVERY'
    while True:
        match_all_frozen(dataset,store,ownership,config,backend,debug)
        version=ownership.meta('model_version',0)
        try:source=ownership.first_unexplained(version)
        except FrozenMeasurementConflict as error:
            status='UNEXPLAINED_OBSERVATION_HAS_FROZEN_MEASUREMENT_OWNER'
            json_write(root/'discovery/frozen_measurement_conflict.json',{'source':error.reference.record(dataset),'owner':error.owner,'reason':str(error),'old_trajectories_changed':False})
            break
        if source is None:break
        if max_events is not None and completed>=max_events:
            status='EXPLICIT_TEST_EVENT_LIMIT';break
        point_id=ownership.allocate_id(store.n0);before=store.frozen_signature()
        ownership.reserve(source,point_id)
        debug.event_start(point_id,source,version)
        try:track,claims,color=track_one_point(point_id,source,dataset,config,backend,ownership)
        except (ArithmeticError,ValueError,RuntimeError,np.linalg.LinAlgError) as error:
            ownership.release_reservations(point_id);ownership.fail(source,version,str(error));debug.event_failure(point_id,source,version,str(error));failures+=1;continue
        path,record=store.save_new(point_id,track,source,color)
        ownership.claim(claims,kind=1,point_record=(point_id,path,record))
        store.append(path,record);store.assert_old_frozen(before);completed+=1
        debug.event_complete(point_id,source,version+1,track,store,ownership)
        # No motion confirmation, global score, old-track fit or smoother.
        # The next iteration restarts the authoritative reverse scan at T.
    debug.final(store,ownership,status,completed,failures,time.monotonic()-start)
    return store,ownership
