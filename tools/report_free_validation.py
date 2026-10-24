#!/usr/bin/env python3
"""Independently inspect completed source/output records; never promote unresolved inference."""
import argparse,json,sys
from pathlib import Path
import numpy as np
REPO=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO))
from pointwise_rgbd.dataset import Dataset
from rgbd_common.array_store import ArrayStore


def mask_projection(x,frame):
    """Nominal model hits only: never consult raw background depth/RGB."""
    camera=frame.camera;r=camera.world_to_camera[:3,:3];t=camera.world_to_camera[:3,3]
    q=((r[:,0]*x[:,0,None]+r[:,1]*x[:,1,None])+r[:,2]*x[:,2,None])+t
    uv=q[:,:2]/q[:,2,None]*camera.intrinsics[:2]+camera.intrinsics[2:];pixel=np.floor(np.nan_to_num(uv)+.5).astype(np.int64)
    inside=np.isfinite(q).all(1)&(q[:,2]>camera.near)&(q[:,2]<camera.far)&(pixel[:,0]>=0)&(pixel[:,0]<camera.width)&(pixel[:,1]>=0)&(pixel[:,1]<camera.height)
    ids=np.flatnonzero(inside);linear=pixel[ids,1]*camera.width+pixel[ids,0];order=np.lexsort((ids,q[ids,2],linear))
    _,first=np.unique(linear[order],return_index=True);hits=linear[order[first]];positive=int(frame.mask.ravel()[hits].sum());negative=len(hits)-positive
    return np.array([len(hits),positive,negative if frame.object_only_no_external_occluders is not True else 0,negative if frame.object_only_no_external_occluders is True else 0],dtype=np.int64)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',type=Path,required=True);args=parser.parse_args();root=args.run
    manifest=json.loads((root/'manifest.json').read_text());result=json.loads((root/'result.json').read_text());options=manifest['arguments']
    data=Dataset(options['scene'],options.get('frame_limit'),options.get('camera_limit'),fps=options.get('fps'),dt_seconds=options.get('dt_seconds'),timestamps_file=options.get('timestamps_file'))
    audit={'full_source':result['full_source'],'status':result['status'],'complete':result['complete'],'frames':len(data.times),'cameras':len(data.cameras),'timing':data.base.timing_report,'mask_visibility_premise':data.base.metadata_report['mask_visibility_premise'],'directions':{},'no_gt':'No part, joint or GT object pose values consumed by the estimator; this audit reads only timing/calibration/foreground observations.'}
    for name in ('first','last'):
        if name not in result['directions']:continue
        local=root/name;phase=local/'phase1'
        if options.get('phase1_from'):phase=Path(options['phase1_from'])/name/'phase1'
        summary=json.loads((phase/'summary.json').read_text());run=json.loads((phase/'run.json').read_text());records=[json.loads(p.read_text()) for p in sorted((phase/'filter').glob('*.json'))]
        states=np.load(phase/'state.npy',mmap_mode='r');cov=np.load(phase/'covariance.npy',mmap_mode='r');source=np.load(phase/'partition.npz')['source'];anchor=run['anchor_frame']
        if len(records)!=len(data.times)-1:raise AssertionError('Incomplete temporal pass')
        for ci in np.unique(source[:,1]):
            if not data.observation(anchor,int(ci)).valid.ravel()[source[source[:,1]==ci,2]].all():raise AssertionError('Background entered endpoint catalogue')
        minimum=float('inf');asym=0.;mask_counts=np.zeros(4,np.int64)
        for k in range(len(data.times)):
            if not np.isfinite(states[k]).all():raise AssertionError('Nonfinite state')
            c=np.asarray(cov[k]);minimum=min(minimum,float(np.linalg.eigvalsh(c).min()));asym=max(asym,float(abs(c-c.transpose(0,2,1)).max()))
            for ci in range(len(data.cameras)):mask_counts+=mask_projection(np.asarray(states[k,:,:3]),data.observation(k,ci))
        if run.get('camera_policy')!='all_available' or run.get('production_camera_indices')!=list(range(len(data.cameras))):raise AssertionError('Production excluded available cameras')
        if any([v['camera_index'] for v in r['association']]!=list(range(len(data.cameras))) for r in records):raise AssertionError('A frame skipped an available camera')
        store=ArrayStore(phase/'factors.sqlite');refs=set();counts=np.zeros(len(data.cameras),np.int64);squared=np.zeros((len(data.cameras),3))
        for k in range(len(data.times)):
            if k==anchor:continue
            row=store.read(str(k))
            for ci in np.unique(row['camera']):
                use=row['camera']==ci;pixels=row['pixel'][use]
                if not data.observation(k,int(ci)).valid.ravel()[pixels].all():raise AssertionError('Background factor')
                identities={(k,int(ci),int(p)) for p in pixels}
                if len(identities)!=len(pixels):raise AssertionError('Repeated independent assimilation of one raw return')
                refs|=identities;counts[int(ci)]+=len(pixels);squared[int(ci)]+=(row['residual'][use]**2).sum(0)
        store.close()
        coupling=ArrayStore(phase/'cross_point_coupling.sqlite');last=coupling.read(str(run['order'][-1]));coupling.close()
        record={**summary,'minimum_point_covariance_eigenvalue':minimum,'maximum_covariance_asymmetry':asym,'finite_all_states':True,'foreground_only_catalogue_and_all_raw_factors':True,'camera_policy':'all_available','production_camera_indices':list(range(len(data.cameras))),'raw_factors_by_camera':counts.tolist(),'uvz_rmse_by_camera_pixels_pixels_metres':np.sqrt(squared/np.maximum(counts[:,None],1)).tolist(),'unique_raw_returns':len(refs),'max_true_covariance_solve_relative_residual':max(v['max_relative_residual'] for r in records for v in r['covariance_linear_solve']),'confirmed_hard_mask_violations':sum(r['remaining_mask_violations'] for r in records),'negative_mask_scope':run['negative_mask_support'],'closed_covariance_saved_shapes':{k:list(v.shape) for k,v in last.items()}}
        record.pop('frame_diagnostics',None)
        record['nominal_mask_projection_counts']=dict(zip(('first_hits','mask_positive','mask_negative_undetermined','mask_negative_hard_contradictions'),map(int,mask_counts)))
        record['group_iterations_limit_count']=sum(int(np.sum(r['iteration_limit_reached'])) for r in records)
        record['group_line_search_stops_count']=sum(int(np.sum(r['line_search_stops'])) for r in records)
        record['maximum_group_energy_increase']=max(float(np.max(np.asarray(r['after'])-r['before'])) for r in records)
        for stage in ('coherence','rigid','joint','support'):
            path=local/stage/'result.json'
            if path.exists():
                r=json.loads(path.read_text());record[stage]={k:r[k] for k in ('status','point_count','representative_nodes','complete_multiview','endpoint_evidence','candidates','labels','converged','reason','selected_hypothesis') if k in r}
            else:record[stage]={'status':'NOT_EXECUTED'}
        rigid_root=local/'rigid'
        if (rigid_root/'factors.sqlite').exists():
            rigid=ArrayStore(rigid_root/'factors.sqlite');camera_counts=np.zeros(len(data.cameras),np.int64)
            for k in range(len(data.times)):
                if not rigid.contains(str(k)):continue
                row=rigid.read(str(k))
                for ci in np.unique(row['camera']):
                    use=row['camera']==ci;pixel=row['pixel'][use]
                    if not data.observation(k,int(ci)).valid.ravel()[pixel].all():raise AssertionError('Background in shared rigid factors')
                    if len(np.unique(pixel))!=len(pixel):raise AssertionError('Duplicate rigid raw assimilation')
                    camera_counts[int(ci)]+=len(pixel)
                if k==anchor:
                    prior_refs={(int(ci),int(pixel)) for _,ci,pixel in source[np.load(local/'coherence/representatives.npz')['moving_support']]}
                    if prior_refs & {(int(ci),int(pixel)) for ci,pixel in zip(row['camera'],row['pixel'])}:raise AssertionError('Endpoint prior reused as a likelihood')
            rigid.close();record['rigid']['foreground_only_raw_factors']=True
            record['rigid']['raw_factors_by_camera']=camera_counts.tolist()
            initialization=json.loads((rigid_root/'initial_rigid_fit.json').read_text())
            if any(r.get('processed_camera_indices')!=list(range(len(data.cameras))) for r in initialization['frames']):raise AssertionError('Rigid initialization excluded a camera')
            record['rigid']['all_cameras_attempted_at_every_frame']=True
        audit['directions'][name]=record
    (root/'reports').mkdir(exist_ok=True)
    (root/'reports/independent_output_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    text=['# Actual dataset and output audit',f"Status: {audit['status']}; full source: {audit['full_source']}",f"{audit['frames']} frames, {audit['cameras']} cameras; timing source: {audit['timing']['time_source']}; FPS: {audit['timing']['fps']}",'All saved endpoint points and Phase-1 raw factors were rechecked against actual foreground validity. Source/frame IDs and all-camera participation was checked, not inferred from filenames.',f"Mask premise: {audit['mask_visibility_premise']}",'Undeclared external-occluder premise means negative support is undetermined, not proven feasible. No background values were used to resolve that uncertainty.','See `independent_output_audit.json` for numerical covariance, matching, model and convergence results.']
    (root/'reports/dataset_audit.md').write_text('\n\n'.join(text)+'\n')
    print(json.dumps({'status':audit['status'],'frames':audit['frames'],'cameras':audit['cameras'],'fps':audit['timing']['fps'],'directions':{k:{f:v[f] for f in ('foreground_only_catalogue_and_all_raw_factors','minimum_point_covariance_eigenvalue','max_true_covariance_solve_relative_residual','nominal_mask_projection_counts')} for k,v in audit['directions'].items()}}))


if __name__=='__main__':main()
