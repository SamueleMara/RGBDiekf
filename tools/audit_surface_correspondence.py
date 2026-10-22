#!/usr/bin/env python3
"""Compare frozen-cache coverage policies. NEVER filters or reoptimizes a point."""
import argparse,csv,hashlib,io,json,resource,shutil,sys,time,traceback
from pathlib import Path
import numpy as np
from tqdm.auto import tqdm
REPO=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO))
from pointwise_rgbd.frozen import FrozenCache
from pointwise_rgbd.dataset import Dataset
from pointwise_rgbd.observations import samples
from pointwise_rgbd.coverage import SurfaceIndex
from pointwise_rgbd.legacy_coverage_audit import nonexclusive_legacy
from pointwise_rgbd.store import json_write
from pointwise_rgbd.diagnostics import distribution
from rgbd_common.backends.torch_cuda import Backend


def unpack(blob):
    with np.load(io.BytesIO(blob),allow_pickle=False) as record:return {key:record[key] for key in record.files}


def visual(directory,frame,raw,surface,result,baseline):
    from PIL import Image,ImageDraw
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from plotly.offline import get_plotlyjs
    directory.mkdir(parents=True,exist_ok=True)
    masks=[('current_ownership',baseline),('geometric_knn',result.geometry_explained),('geometric_rgb_knn',result.explained)]
    original=np.uint8(np.clip(frame.rgb,0,1)*255)
    figure,axes=plt.subplots(1,3,figsize=(18,6))
    for ax,(name,mask) in zip(axes,masks):
        rgb=original.copy();pixels=rgb.reshape(-1,3)
        tint=np.where(mask[:,None],np.array([0,255,80]),np.array([255,25,60]))
        pixels[raw.pixel]=np.uint8(.4*pixels[raw.pixel]+.6*tint)
        image=Image.fromarray(rgb);draw=ImageDraw.Draw(image)
        # Transported cached means, not rendered or newly optimized geometry.
        h=surface.state[:,:3]@frame.camera.world_to_camera[:3,:3].T+frame.camera.world_to_camera[:3,3]
        uv=h[:,:2]/h[:,2,None]*frame.camera.intrinsics[:2]+frame.camera.intrinsics[2:]
        for index in np.linspace(0,len(h)-1,min(1200,len(h)),dtype=int):
            x,y=uv[index]
            if h[index,2]>0 and np.isfinite([x,y]).all() and 0<=x<frame.camera.width and 0<=y<frame.camera.height:
                draw.ellipse((x-1,y-1,x+1,y+1),fill='#00bfff')
        image.save(directory/(name+'.png'));ax.imshow(image);ax.set_title(f'{name}: {int(mask.sum())}/{len(mask)} explained');ax.axis('off')
    figure.suptitle(f'Actual frame {frame.k}, camera {frame.camera.name}; green coverage / red unexplained / cyan frozen XYZ display sample')
    figure.tight_layout();figure.savefig(directory/'comparison.png');plt.close(figure)
    # Sampling is solely an explicitly labeled browser payload limit.
    display=np.linspace(0,len(surface.ids)-1,min(4096,len(surface.ids)),dtype=int)
    traces=[{'type':'scatter3d','mode':'markers','name':'Transported SOLVED (display sample)','x':surface.state[display,0].tolist(),'y':surface.state[display,1].tolist(),'z':surface.state[display,2].tolist(),'marker':{'size':2,'color':'#5684a0'}}]
    for name,mask,color in [('Geometrically + RGB explained',result.explained,'#00bb55'),('Unexplained under declared tests',~result.explained,'#ff3355')]:
        ids=np.flatnonzero(mask)
        if len(ids)>4096:ids=ids[np.linspace(0,len(ids)-1,4096,dtype=int)]
        xyz=raw.xyz[ids];traces.append({'type':'scatter3d','mode':'markers','name':name+' (display sample)','x':xyz[:,0].tolist(),'y':xyz[:,1].tolist(),'z':xyz[:,2].tolist(),'marker':{'size':3,'color':color}})
    asset=directory.parent/'plotly.min.js'
    if not asset.exists():asset.write_text(get_plotlyjs())
    page='<!doctype html><html><head><meta charset="utf-8"><script src="../plotly.min.js"></script></head><body><h2>World-coordinate cached surface correspondence</h2><p>Actual frozen XYZ and raw backprojected RGB-D. Display sampled only; ALL foreground observations used in numerical audit. Toggle traces in legend; rotate/zoom freely. Unexplained means incompatible under this finite-k, uncalibrated policy, not proof of missing physical geometry.</p><div id="plot" style="height:85vh"></div><script>Plotly.newPlot("plot",DATA,{scene:{aspectmode:"data",xaxis:{title:"world X (m)"},yaxis:{title:"world Y (m)"},zaxis:{title:"world Z (m)"}})</script></body></html>'
    (directory/'world_coverage.html').write_text(page.replace('DATA',json.dumps(traces,allow_nan=False)))
    fig=plt.figure(figsize=(9,7));ax=fig.add_subplot(111,projection='3d')
    for trace in traces:ax.scatter(trace['x'],trace['y'],trace['z'],s=2,c=trace['marker']['color'],label=trace['name'])
    ax.set(xlabel='world X (m)',ylabel='world Y (m)',zlabel='world Z (m)',title='Actual HTML data, display sampling only');ax.legend(fontsize=7);fig.tight_layout();fig.savefig(directory/'world_coverage.preview.png');plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--cache',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--frames',type=int,nargs='+',help='Explicit audit subset of temporal indices; default every cached frame')
    parser.add_argument('--device',choices=['cpu','cuda'],default='cuda');parser.add_argument('--visual-frames',type=int,nargs='+',default=[0,1,38,39,80,160,240,327])
    args=parser.parse_args();root=args.output.resolve();cache_root=args.cache.resolve()
    if root==cache_root or cache_root in root.parents:parser.error('Audit output must be separate from immutable cache')
    if root.exists() and any(root.iterdir()):parser.error('Preserving existing audit; use a new output directory')
    root.mkdir(parents=True,exist_ok=True);start=time.monotonic()
    try:
        cache=FrozenCache(cache_root);before=cache.frozen_signature();config=cache.config();backend=Backend(args.device)
        data=Dataset(cache.manifest['dataset']['input'],**cache.manifest['dataset']['test_limits'])
        if data.frame_ids!=cache.manifest['dataset']['original_frame_ids'] or not np.array_equal(data.times,cache.manifest['dataset']['timestamps_seconds']):raise ValueError('Cache/dataset timing or frame identities changed')
        cameras=data.manifest()['cameras'];digest=hashlib.sha256(json.dumps(cameras,sort_keys=True).encode()).hexdigest()
        if digest!=cache.manifest['calibration_sha256']:raise ValueError('Cached calibration differs from selected dataset')
        fingerprints=json.loads((cache_root/'raw_observations/input_fingerprints.json').read_text())
        frames=list(range(len(data.times))) if args.frames is None else sorted(set(args.frames))
        if any(k<0 or k>=len(data.times) for k in frames):raise ValueError('Audit frame outside cache')
        manifest={'task':'surface correspondence audit only','cache':str(cache_root),'frames':frames,'cameras':len(data.cameras),'tail_probability':config.association_tail_probability,'neighbors':config.neighbors,'geometry_covariance':'2*(P_position + backprojected_observation_covariance): unknown-cross-covariance PSD upper bound','appearance_covariance':'4*R_rgb: unknown-cross-covariance PSD upper bound','rank':'sum of marginal Gaussian negative log likelihoods (composite score); no adjustable geometry/color weight','appearance':'whitened RGB cosine/angular contribution + whitened brightness, exactly equal to one RGB Mahalanobis distance; raw RGB cosine reported','gates':{str(d):config.gate(d) for d in (1,2,3)},'gate_qualification':'1% per component, uncalibrated; multiple dependent tests do not imply 99% familywise coverage; no retuning','no_optimization_invoked':True,'source_hashes':{str(p.relative_to(REPO)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),REPO/'pointwise_rgbd/coverage.py',REPO/'pointwise_rgbd/legacy_coverage_audit.py',REPO/'pointwise_rgbd/frozen.py',REPO/'pointwise_rgbd/association.py']}}
        json_write(root/'manifest.json',manifest);(root/'raw_classification').mkdir();(root/'distributions').mkdir()
        metrics=('nearest_distance_m','geometry_selected_distance_m','selected_distance_m','nearest_rgb_cosine','geometry_rgb_cosine','rgb_cosine')
        streams={name:(root/'distributions'/f'{name}.float64').open('wb') for name in metrics}
        rows=[];offset=0
        for k in tqdm(frames,desc='Audit cached correspondences (NO filtering)',unit='frame'):
            state,p,color,ids,_=cache.frame(k);cache.verify_frame(k,state,p);surface=SurfaceIndex(state,p,color,ids)
            for ci in range(len(data.cameras)):
                frame=data.observation(k,ci)
                if fingerprints[f'{k:06d}_{ci:03d}']!=data.base.observation_fingerprints[f'{k:06d}_{ci:03d}']:raise ValueError(f'Original cached RGB-D/mask input changed at frame {k}, camera {ci}')
                raw=samples(frame,config)
                blob=cache.owner_db.execute('SELECT data FROM owners WHERE frame=? AND camera=?',(k,ci)).fetchone()[0];old=unpack(blob)
                owner=old['owner'][raw.pixel];kind=old['kind'][raw.pixel];baseline=owner>=0;actual_owner=np.where(np.isin(kind,[1,3]),owner,-1)
                result=surface.match(frame,config,backend,raw)
                legacy,tied,witness,old_stats=nonexclusive_legacy(surface,frame,raw,old['owner'],config,backend)
                g=result.geometry_explained;a=result.explained;unmatched=~baseline
                row={'frame':k,'original_frame':data.frame_ids[k],'camera':ci,'raw_valid':len(raw.pixel),'current_explained':int(baseline.sum()),'current_unmatched':int(unmatched.sum()),'legacy_nonexclusive_explained':int(legacy.sum()),'legacy_same_tie_explained':int(tied.sum()),'geometry_explained':int(g.sum()),'appearance_explained':int(a.sum()),'current_unmatched_legacy_nonexclusive':int((unmatched&legacy).sum()),'current_unmatched_legacy_same_ties':int((unmatched&tied).sum()),'strict_owned_point_witness':int((unmatched&witness).sum()),'current_unmatched_geometry_explained':int((unmatched&g).sum()),'current_unmatched_appearance_explained':int((unmatched&a).sum()),'current_explained_geometry_rejected':int((baseline&~g).sum()),'current_explained_appearance_rejected':int((baseline&~a).sum()),'geometry_supported_color_rejected':int((g&~a).sum()),'unexplained_with_frozen_measurement_owner':int(((actual_owner>=0)&~a).sum()),'unowned_unexplained_seed_candidates':int(((actual_owner<0)&~a).sum()),'legacy_seconds':old_stats.get('seconds',0.),**result.stats}
                for prefix,count in [('current',row['current_explained']),('geometry',row['geometry_explained']),('appearance',row['appearance_explained'])]:row[prefix+'_explained_fraction']=count/len(raw.pixel) if len(raw.pixel) else None
                rows.append(row)
                with (root/'by_frame_camera.jsonl').open('a') as stream:stream.write(json.dumps(row)+'\n')
                values={name:getattr(result,name) for name in metrics}
                for name,value in values.items():streams[name].write(np.asarray(value,dtype=np.float64).tobytes())
                np.savez_compressed(root/'raw_classification'/f'{k:06d}_{ci:03d}.npz',pixel=raw.pixel,baseline_owner=owner,baseline_kind=kind,legacy_nonexclusive=legacy,legacy_same_ties=tied,assigned_point_witness=witness,geometry_explained=g,appearance_explained=a,geometry_point_id=result.geometry_point_id,appearance_point_id=result.point_id)
                json_write(root/'distributions'/f'{k:06d}_{ci:03d}.json',{name:distribution(value) for name,value in values.items()})
                if k in args.visual_frames:visual(root/'visuals'/f'{k:06d}_{ci:03d}',frame,raw,surface,result,baseline)
                with (root/'metric_offsets.jsonl').open('a') as stream:stream.write(json.dumps({'frame':k,'camera':ci,'offset':offset,'count':len(raw.pixel)})+'\n')
                offset+=len(raw.pixel)
            cache.assert_old_frozen(before)
        for stream in streams.values():stream.close()
        numeric=[key for key,value in rows[0].items() if isinstance(value,(int,float)) and key not in ('frame','original_frame','camera','tail_probability','solved') and not key.endswith('_fraction')]
        totals={key:sum(row[key] for row in rows) for key in numeric}
        total=totals['raw_valid'];totals['raw_valid']=total
        for prefix in ('current','geometry','appearance'):totals[prefix+'_explained_fraction']=totals[prefix+'_explained']/total if total else None
        if args.frames is None:
            original=json.loads((cache_root/'final/summary.json').read_text())
            if totals['current_unmatched']!=original['unmatched_raw_observations']:raise AssertionError('Baseline is not the original reported unmatched set')
        for row in rows:
            if row['frame']==0 and (row['geometry_explained']!=row['raw_valid'] or row['appearance_explained']!=row['raw_valid']):raise AssertionError('Frame-zero all-source coverage failed')
        global_distributions={}
        for name in metrics:
            values=np.memmap(root/'distributions'/f'{name}.float64',mode='r',dtype=np.float64,shape=(total,))
            global_distributions[name]=distribution(values)
        json_write(root/'global_distributions.json',global_distributions)
        with (root/'before_after_by_frame_camera.csv').open('w') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
        cache.assert_old_frozen(before)
        for record in cache.added:
            if hashlib.sha256(Path(record['path']).read_bytes()).hexdigest()!=record['sha256']:raise AssertionError('Old appended point changed')
        summary={'status':'AUDIT_COMPLETE','all_cached_frames':args.frames is None,'frames_checked':len(frames),'cameras_checked':len(data.cameras),'initial_points':cache.n0,'cached_SOLVED':cache.count,'totals':totals,'frozen_initial_digests_verified':True,'frozen_appended_tracks_verified':True,'optimizer_calls':0,'runtime_seconds':time.monotonic()-start,'peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,'gpu':backend.memory(),'qualification':'Compatibility under the declared finite-k, covariance-bound, uncalibrated tests; not ground-truth physical identity or an exact global correspondence posterior.'}
        json_write(root/'summary.json',summary);print(json.dumps(summary,indent=2));return 0
    except BaseException:
        (root/'failure.log').write_text(traceback.format_exc());raise


if __name__=='__main__':raise SystemExit(main())
