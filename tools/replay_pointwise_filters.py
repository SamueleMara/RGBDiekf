#!/usr/bin/env python3
"""Replay persisted real factors to verify point means/P and recover solver logs."""
import argparse,csv,json,sys
from pathlib import Path
import numpy as np
import torch
REPO=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO))
from pointwise_rgbd.config import Config
from pointwise_rgbd.dataset import Dataset
from pointwise_rgbd.observations import initialize_seed
from pointwise_rgbd.state6 import ObservationRef
from pointwise_rgbd.process6 import predict
from pointwise_rgbd.filter_iekf import correct
from pointwise_rgbd.store import json_write
from rgbd_common.backends.torch_cuda import Backend


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);parser.add_argument('--device',default='cuda',choices=['cuda','cpu']);args=parser.parse_args()
    manifest=json.loads((args.run/'run_manifest.json').read_text());cfg=manifest['config']
    for name in ('qc','qx','velocity_covariance','pixel_covariance','color_covariance'):cfg[name]=np.array(cfg[name])
    config=Config(**cfg);dataset=Dataset(manifest['dataset']['input'],**manifest['dataset']['test_limits']);backend=Backend(args.device)
    index=json.loads((args.run/'final/point_tracks.json').read_text());result=[]
    args.output.mkdir(parents=True,exist_ok=True)
    for record in index['appended']:
        with np.load(record['path']) as saved:
            source=ObservationRef(*saved['source']);refs=saved['measurement_refs'].copy();state=saved['state'].copy();p=saved['covariance'].copy()
        anchor,anchor_p,_=initialize_seed(source,dataset,config);rows=[];max_state=0.;max_p=0.;bitwise=True
        if not np.array_equal(anchor,state[source.frame]) or not np.array_equal(anchor_p,p[source.frame]):raise AssertionError('Anchor changed on replay')
        for name,indices,reverse in [('backward',range(source.frame-1,-1,-1),True),('forward',range(source.frame+1,len(dataset.times)),False)]:
            current=backend.tensor(anchor[None]).clone();cov=backend.tensor(anchor_p[None]).clone();previous=source.frame
            for k in indices:
                prior,pminus=predict(current,cov,abs(dataset.times[k]-dataset.times[previous]),config,reverse);matches=[]
                for ci,camera in enumerate(dataset.cameras):
                    selected=refs[(refs[:,0]==k)&(refs[:,1]==ci),2]
                    if len(selected)>1:raise AssertionError('Duplicate camera factor')
                    z=np.empty((len(selected),3));r=np.empty((len(selected),3,3))
                    if len(selected):
                        frame=dataset.observation(k,ci);pixel=int(selected[0]);y,x=divmod(pixel,camera.width)
                        if not frame.valid[y,x]:raise AssertionError('Replay input is background/invalid')
                        z[0]=[x,y,frame.depth[y,x]];r[0]=config.measurement_covariance(z[0,2])
                    matches.append({'point':backend.tensor(np.zeros(len(selected),np.int64),torch.long),'camera':camera,'z':backend.tensor(z),'R':backend.tensor(r)})
                current,cov,report=correct(prior,pminus,matches,config)
                actual=current.cpu().numpy()[0];actual_p=cov.cpu().numpy()[0]
                max_state=max(max_state,float(abs(actual-state[k]).max()));max_p=max(max_p,float(abs(actual_p-p[k]).max()))
                bitwise &= np.array_equal(actual,state[k]) and np.array_equal(actual_p,p[k])
                np.testing.assert_allclose(actual,state[k],rtol=1e-12,atol=1e-12);np.testing.assert_allclose(actual_p,p[k],rtol=1e-12,atol=1e-12)
                rows.append({'frame':k,'original_frame':dataset.frame_ids[k],'direction':name,'iterations':int(report['iterations'][0]),'energy_before':float(report['before'][0]),'energy_after':float(report['after'][0]),'line_search_stops':int(report['line_search_stops'][0]),'iteration_limit_reached':bool(report['iteration_limit_reached'][0]),'raw_measurements':int(report['measurement_count'][0])});previous=k
        json_write(args.output/f"point_{record['point_id']:012d}_optimizer_replay.json",rows)
        result.append({'point_id':record['point_id'],'steps_replayed':len(rows),'state_max_absolute_difference':max_state,'covariance_max_absolute_difference':max_p,'bitwise_identical':bool(bitwise),'all_final_nonlinear_energies_nonincreasing':all(r['energy_after']<=r['energy_before'] for r in rows),'iteration_limit_steps':sum(r['iteration_limit_reached'] for r in rows)})
    with (args.output/'raw_pixel_counts.csv').open('w') as stream:
        fields=['frame','original_frame','camera','raw_pixels','foreground_pixels','invalid_foreground_pixels','valid_foreground'];writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
        for k in range(len(dataset.times)):
            for ci in range(len(dataset.cameras)):
                frame=dataset.observation(k,ci);valid=int(frame.valid.sum());fg=int(frame.mask.sum())
                writer.writerow(dict(zip(fields,[k,dataset.frame_ids[k],ci,frame.depth.size,fg,fg-valid,valid])))
    report={'status':'PASS','run':str(args.run),'device':args.device,'tracks':result,'qualification':'Replay of the actual persisted factors, not reselection of correspondences or a new reconstruction. Raw counts are derived from actual input masks/validity.'}
    json_write(args.output/'replay_summary.json',report);print(json.dumps(report,indent=2))


if __name__=='__main__':main()
