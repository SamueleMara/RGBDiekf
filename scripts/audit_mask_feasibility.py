"""Report real depth-order probabilities without inventing a decision policy."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from rgbd_common.backends.torch_cuda import Backend,numpy
from rgbd_common.config import load_config
from rgbd_common.dataloader import load_dataset
from rgbd_common.geometry import PointMap
from rgbd_common.projection_cuda import transport_and_project,build_pixel_bins_and_zbuffer


def main():
    p=argparse.ArgumentParser();p.add_argument('--input',required=True,type=Path);p.add_argument('--map',required=True,type=Path);p.add_argument('--output',required=True,type=Path);a=p.parse_args()
    if a.output.exists():raise FileExistsError('Choose a fresh audit directory')
    a.output.mkdir(parents=True);data=load_dataset(a.input);model=PointMap.load(a.map);config=load_config();backend=Backend('cuda');rows=[]
    for ci,camera in enumerate(data.cameras):
        frame=data.observation(0,ci);projected=transport_and_project(backend.tensor(model.xyz,backend.input_dtype),backend.tensor(model.group,torch.long),backend.tensor(np.tile(np.eye(4),(int(model.group.max())+1,1,1)),backend.input_dtype),camera)
        bins=build_pixel_bins_and_zbuffer(projected,camera);ids=bins['visible_ids'];pixel=projected['pixel'][ids,1]*camera.width+projected['pixel'][ids,0];host=numpy(pixel)
        depth=np.asarray(frame.depth).ravel()[host];background=(~frame.mask.ravel()[host])&np.isfinite(depth)&(depth>0)
        ids=ids[backend.tensor(background,torch.bool)];pixel=pixel[backend.tensor(background,torch.bool)]
        observed=backend.tensor(depth[background]);z=projected['depth'][ids].double();variance=config.rgbd_covariance[3,3]+observed**2*config.rgbd_depth_squared[3,3]
        standard=(observed-z)/torch.sqrt(variance);log_visible=torch.special.log_ndtr(standard);log_occluded=torch.special.log_ndtr(-standard)
        np.savez(a.output/f'background_depth_order_frame_000000_cam_{ci:03d}.npz',point_id=model.point_id[numpy(ids)],pixel=numpy(pixel),predicted_depth_m=numpy(z),observed_background_depth_m=numpy(observed),depth_variance_m2=numpy(variance),log_probability_model_is_closer=numpy(log_visible),log_probability_observation_is_closer=numpy(log_occluded))
        row={'camera':ci,'front_projected_background_valid':len(ids),'probability_model_is_closer_quantiles':np.quantile(numpy(torch.exp(log_visible)),[0,.25,.5,.75,1]).tolist() if len(ids) else [],'normalized_depth_order_quantiles':np.quantile(numpy(standard),[0,.25,.5,.75,1]).tolist() if len(ids) else []};rows.append(row)
    report={'purpose':'Preserved frame-0 map diagnostic, not reconstruction or acceptance','map_file':str(a.map),'map_points':model.count,'frame':0,'camera_count':len(data.cameras),'visibility_decision':'UNSPECIFIED: probabilities only; no distance cutoff or confidence level was assumed','noise_source':str(config.source),'noise_class':'documented-initial-guess','by_camera':rows,'gpu':backend.memory()}
    (a.output/'mask_depth_order_audit.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':main()
