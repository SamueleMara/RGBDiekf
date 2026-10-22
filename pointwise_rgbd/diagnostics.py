from pathlib import Path
import csv,json,time,resource
import numpy as np
from .store import json_write
from .visualization import trajectory_viewer,event_plots,image_overlays,plotly_asset


def distribution(values):
    x=np.asarray(values,dtype=float);x=x[np.isfinite(x)]
    if not len(x):return {'count':0,'quantiles':None,'histogram':None}
    hist,edges=np.histogram(x,bins=32)
    return {'count':len(x),'quantiles':dict(zip(('min','median','p90','p95','p99','max'),np.quantile(x,[0,.5,.9,.95,.99,1]).tolist())),'histogram':{'count':hist.tolist(),'edges':edges.tolist()}}


def append_csv(path,record):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);exists=path.exists()
    with path.open('a') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(record));
        if not exists:writer.writeheader()
        writer.writerow(record)


class Diagnostics:
    def __init__(self,root,dataset,config,backend):
        self.root=Path(root);self.dataset=dataset;self.config=config;self.backend=backend;self.started=time.monotonic()
        for name in ('initial_forward','discovery/events','final'): (self.root/name).mkdir(parents=True,exist_ok=True)
        self.frame_records=[]
        path=self.root/'initial_forward/summary.json'
        if path.exists():self.frame_records=json.loads(path.read_text()).get('frame_records',[])
    def event(self,record):
        with (self.root/'discovery/events.jsonl').open('a') as stream:stream.write(json.dumps(record,allow_nan=False)+'\n')
    def initial_anchor(self,store):
        record={'frame':0,'original_frame':self.dataset.frame_ids[0],'points':store.n0,'source_observations':store.n0,'observed':store.n0,'semantics':'Every original frame-0 raw valid foreground pixel retained without downsampling; anchor posterior initialized once, no second anchor update'}
        json_write(self.root/'initial_forward/summary.json',record)
    def initial_frame(self,k,statistics,nis,counts,p,filter_summary,filter_seconds,elapsed):
        matched=np.concatenate([np.asarray(r['matched_d2']) for r in statistics]);unmatched=np.concatenate([np.asarray(r['unmatched_d2']) for r in statistics])
        record={'frame':k,'original_frame':self.dataset.frame_ids[k],'time_seconds':float(self.dataset.times[k]),'real_observed_points':int((counts>0).sum()),'raw_matches':int(counts.sum()),'prediction_only_points':int((counts==0).sum()),'nis':distribution(nis),'matched_d2':distribution(matched),'unmatched_d2':distribution(unmatched),'position_covariance_trace_m2':distribution(np.trace(p[:,:3,:3],axis1=1,axis2=2)),'filter':filter_summary,'filter_seconds':filter_seconds,'frame_seconds':elapsed,'knn_seconds':sum(r['knn_seconds'] for r in statistics),'projection_seconds':sum(r['projection_seconds'] for r in statistics),'covariance_carry_forward':'previous posterior through F P F^T + Q; no reset'}
        self.frame_records.append(record)
        record['index_build_seconds']=sum(r['index_build_seconds'] for r in statistics)
        record['association_seconds']=sum(r['association_seconds'] for r in statistics)
        record['matched_depth_residual_m']=distribution(np.concatenate([np.asarray(r['matched_depth_residual_m']) for r in statistics]))
        record['ambiguity_count']=sum(r['ambiguous'] for r in statistics)
        record['ownership_conflicts']=sum(r['conflicts'] for r in statistics)
        record['invalid_foreground_pixels']=sum(r['invalid_foreground_pixels'] for r in statistics)
        append_csv(self.root/'initial_forward/nis_by_frame.csv',{'frame':k,'timestamp_seconds':float(self.dataset.times[k]),'observed':record['real_observed_points'],'raw_matches':record['raw_matches'],'nis_median':None if not record['nis']['quantiles'] else record['nis']['quantiles']['median'],'nis_p99':None if not record['nis']['quantiles'] else record['nis']['quantiles']['p99'],'min_P_eigenvalue':filter_summary['min_eigenvalue'],'max_P_condition':filter_summary['max_condition'],'filter_seconds':filter_seconds})
        json_write(self.root/'initial_forward/summary.json',{'status':'INITIAL_FORWARD_RUNNING','frame_records':self.frame_records})
    def initial_complete(self,store,ownership):
        summary={'status':'INITIAL_FORWARD_FROZEN','N0':store.n0,'frames':len(self.dataset.times),'cameras':len(self.dataset.cameras),'initial_forward_invocations':ownership.meta('initial_invocations'),'frame_records':self.frame_records,'state':'[world_xyz, world_velocity], R6; full 6x6 covariance','source_policy':'all frame-0 valid foreground raw samples retained; no optional cross-camera fusion performed','covariance':'conditional on calibration, chosen associations and per-point approximation; no full marginalized joint claim'}
        json_write(self.root/'initial_forward/summary.json',summary)
        json_write(self.root/'initial_forward/matching_histogram.json',[{'frame':r['frame'],'matched':r['matched_d2'],'unmatched':r['unmatched_d2']} for r in self.frame_records])
        script='../assets/plotly.min.js';plotly_asset(self.root)
        points=[{'x':[r['time_seconds'] for r in self.frame_records],'y':[r['raw_matches'] for r in self.frame_records],'type':'scatter','name':'actual matched raw pixels'}]
        (self.root/'initial_forward/diagnostics.html').write_text('<html><head><script src="'+script+'"></script></head><body><h2>Measured initial forward diagnostics</h2><div id="plot"></div><script>Plotly.newPlot("plot",'+json.dumps(points)+',{xaxis:{title:"Seconds"},yaxis:{title:"Actual raw RGB-D matches"}})</script></body></html>')
        trajectory_viewer(self.root/'initial_forward/trajectory_viewer.html',store,self.dataset,self.root)
    def matching_row(self,version,stats,owners):
        free=stats['valid_raw']-int((owners>=0).sum())
        append_csv(self.root/'discovery/matching_by_frame.csv',{'model_version':version,'frame':stats['frame'],'camera':stats['camera'],'raw_pixels':stats['raw_pixels'],'foreground_pixels':stats['foreground_pixels'],'invalid_foreground_pixels':stats['invalid_foreground_pixels'],'valid_foreground':stats['valid_raw'],'owned':int((owners>=0).sum()),'unmatched':free,'new_geometric_matches':stats['matched'],'ambiguous':stats['ambiguous'],'conflicts':stats['conflicts'],'occluded':stats['occluded']})
        m=distribution(stats['matched_d2']);u=distribution(stats['unmatched_d2'])
        gap=None if not m['quantiles'] or not u['quantiles'] else u['quantiles']['min']-m['quantiles']['p99']
        append_csv(self.root/'discovery/nearest_residuals.csv',{'model_version':version,'frame':stats['frame'],'camera':stats['camera'],'nearest_metric_quantiles_m':json.dumps(stats['nearest_metric_quantiles_m']),'sample_spacing_quantiles_m':json.dumps(stats['sample_spacing_quantiles_m']),'matched_distribution':json.dumps(m),'unmatched_distribution':json.dumps(u),'unmatched_min_minus_matched_p99':gap,'margin_claim':'not established' if gap is None or gap<=0 else 'empirical selected-sample separation only'})
    def event_start(self,point,source,version):
        self.event({'event':'START','point_id':point,'source':source.record(self.dataset),'model_version':version,'anchor_assimilations':1,'branch_definition':'independent copies of the same anchor posterior; no smoother'})
    def event_failure(self,point,source,version,reason):
        directory=self.root/f'discovery/events/{point:012d}';directory.mkdir(exist_ok=True)
        json_write(directory/'seed.json',{'point_id':point,'source':source.record(self.dataset),'status':'NUMERICAL_FAILURE','reason':reason,'model_version':version})
        from PIL import Image,ImageDraw
        frame=self.dataset.observation(source.frame,source.camera);image=Image.fromarray(np.uint8(np.clip(frame.rgb,0,1)*255));draw=ImageDraw.Draw(image)
        y,x=divmod(source.pixel,frame.camera.width);draw.ellipse((x-6,y-6,x+6,y+6),outline='red',width=2)
        draw.text((3,3),'NUMERICAL FAILURE: no complete trajectory computed',fill='red');image.save(directory/'failure_anchor.png')
        self.event({'event':'FAILURE','point_id':point,'source':source.record(self.dataset),'model_version':version,'reason':reason,'temporary_claims_released':True})
    def event_complete(self,point,source,version,track,store,ownership):
        directory=self.root/f'discovery/events/{point:012d}';directory.mkdir(exist_ok=True)
        json_write(directory/'seed.json',{'point_id':point,'source':source.record(self.dataset),'anchor_xyz_world':track['state'][source.frame,:3],'anchor_state':track['state'][source.frame],'anchor_covariance':track['covariance'][source.frame],'status':'SOLVED_IMMEDIATELY_AFTER_TWO_SIDED_FILTER','model_version':version,'anchor_uses':track['anchor_uses'],'backward_frames':track['backward_frames'],'forward_frames':track['forward_frames'],'observed_frames':int(track['observed'].sum()),'prediction_only_frames':int((~track['observed']).sum()),'old_tracks_changed':False})
        for k,(state,p) in enumerate(zip(track['state'],track['covariance'])):
            eigen=np.linalg.eigvalsh(p)
            append_csv(directory/'state_by_frame.csv',{'frame':k,'original_frame':self.dataset.frame_ids[k],'time_seconds':float(self.dataset.times[k]),**dict(zip(('x','y','z','vx','vy','vz'),state.tolist())),'observed':bool(track['observed'][k]),'P_trace_mixed_units':float(np.trace(p)),'position_trace_m2':float(np.trace(p[:3,:3])),'min_P_eigenvalue':float(eigen[0]),'P_condition':float(eigen[-1]/eigen[0]),'NIS':float(track['nis'][k]),'raw_measurements':int(track['counts'][k])})
        json_write(directory/'optimizer_by_frame.json',track.get('filter_reports',{'status':'not recorded by original run; see independent replay report'}))
        with (directory/'associations.csv').open('w') as stream:
            fields=['frame','original_frame','camera','camera_name','pixel','point_id','reason','z','residual','S','d2'];writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
            for row in track['associations']:
                row={**row,'original_frame':self.dataset.frame_ids[row['frame']],'camera_name':self.dataset.cameras[row['camera']].name}
                writer.writerow({key:json.dumps(value) if isinstance(value,list) else value for key,value in row.items()})
        event_plots(directory,track,self.dataset);image_overlays(directory,track,source,store,self.dataset,self.config)
        trajectory_viewer(directory/'local_3d.html',store,self.dataset,self.root,point)
        trajectory_viewer(self.root/'discovery/geometry_growth.html',store,self.dataset,self.root)
        self.event({'event':'COMMIT','point_id':point,'source':source.record(self.dataset),'model_version':version,'SOLVED':store.count,'real_observed_frames':int(track['observed'].sum()),'prediction_only_frames':int((~track['observed']).sum()),'anchor_uses':1,'optimizer_count_for_seed':1,'old_states_and_covariances_frozen':True,'restart_next_scan_at_frame':len(self.dataset.times)-1})
    def final(self,store,ownership,status,completed,failures,elapsed):
        unmatched=0;owned=0;used=0;rows=[]
        from rgbd_common.array_store import ArrayStore
        unexplained=ArrayStore(self.root/'final/unexplained_observations.sqlite')
        for k in range(len(self.dataset.times)):
            for ci in range(len(self.dataset.cameras)):
                frame=self.dataset.observation(k,ci);owner,kind=ownership.arrays(k,ci);valid=np.flatnonzero(frame.valid.ravel());free=valid[owner[valid]<0]
                unmatched+=len(free);owned+=int((owner>=0).sum());used+=int(((kind==1)|(kind==3)).sum())
                unexplained.write(f'{k}:{ci}',pixel=free,original_frame=np.array(self.dataset.frame_ids[k]),camera=np.array(self.dataset.cameras[ci].name))
                rows.append({'frame':k,'original_frame':self.dataset.frame_ids[k],'camera':ci,'unexplained':len(free)})
        with (self.root/'final/unexplained_observations.csv').open('w') as stream:
            writer=csv.DictWriter(stream,fieldnames=['frame','original_frame','camera','unexplained']);writer.writeheader();writer.writerows(rows)
        unexplained.close()
        if status=='COMPLETE_FINITE_DISCOVERY' and unmatched:status='NO_ELIGIBLE_SEEDS_NUMERICAL_FAILURES'
        limits=getattr(self.dataset,'test_limits',{})
        selected_complete=status=='COMPLETE_FINITE_DISCOVERY'
        if selected_complete and any(v is not None for v in limits.values()):status='EXPLICIT_TEST_SELECTION_COMPLETE'
        summary={'status':status,'complete_discovery':status=='COMPLETE_FINITE_DISCOVERY','initial_points':store.n0,'new_points':len(store.added),'new_points_this_invocation':completed,'SOLVED':store.count,'first_original_frame':self.dataset.frame_ids[0],'last_original_frame':self.dataset.frame_ids[-1],'frames':len(self.dataset.times),'cameras':len(self.dataset.cameras),'owned_raw_observations':owned,'assimilated_raw_observations':used,'unmatched_raw_observations':unmatched,'failures_this_invocation':failures,'initial_forward_invocations':ownership.meta('initial_invocations'),'model_version':ownership.meta('model_version',0),'runtime_seconds_this_invocation':elapsed,'rss_peak_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,'gpu':self.backend.memory(),'track_store':str(store.directory),'new_tracks':[r['path'] for r in store.added],'covariance_semantics':'Conditional per-point Gaussian, no discrete-association or shared-calibration marginalization; reverse is reoriented process factor, not RTS','limitations':['Finite 8-neighbor local candidate family; association is not an exact global posterior','Prediction-only states are estimates, not sensor observations','No guarantee of physical motion identifiability on textureless/symmetric/occluded geometry'],'test_selection':getattr(self.dataset,'test_limits',{})}
        summary['selected_sequence_discovery_complete']=selected_complete
        summary['stage_seconds']={key:sum(r.get(key,0.) for r in self.frame_records) for key in ('frame_seconds','filter_seconds','knn_seconds','projection_seconds','index_build_seconds','association_seconds')}
        summary['stage_seconds']['image_io_this_invocation']=getattr(self.dataset,'image_io_seconds',None)
        summary['initial_ambiguity_count']=sum(r.get('ambiguity_count',0) for r in self.frame_records)
        summary['initial_ownership_conflicts']=sum(r.get('ownership_conflicts',0) for r in self.frame_records)
        summary['provenance_note']='Raw measurement references in initial_forward/trajectory_store/measurement_refs.sqlite and each appended NPZ are immutable; ownership kind=2 is geometric matching only, never an old filter update.'
        if hasattr(self.dataset,'base'):json_write(self.root/'raw_observations/input_fingerprints.json',self.dataset.base.observation_fingerprints)
        json_write(self.root/'final/summary.json',summary);json_write(self.root/'final/point_tracks.json',{'initial':str(store.directory),'appended':store.added})
        if not (self.root/'discovery/geometry_growth.html').exists():trajectory_viewer(self.root/'discovery/geometry_growth.html',store,self.dataset,self.root)
