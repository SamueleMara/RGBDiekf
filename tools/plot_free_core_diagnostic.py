#!/usr/bin/env python3
"""Full-point actual RGB-D versus proposed core; no display subsampling."""
import argparse,json,sys
from pathlib import Path
import numpy as np
REPO=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO))
from pointwise_rgbd.dataset import Dataset
from pointwise_rgbd.config import load_config
from pointwise_rgbd.observations import samples
from pointwise_rgbd.group_evidence_audit import projection_numpy


def main():
 p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--direction',choices=['first','last'],default='first');p.add_argument('--frame',type=int,default=59);args=p.parse_args()
 manifest=json.loads((args.run/'manifest.json').read_text());o=manifest['arguments'];data=Dataset(o['scene'],o['frame_limit'],o['camera_limit'],fps=o['fps'],dt_seconds=o['dt_seconds'],timestamps_file=o['timestamps_file']);config=load_config(o['noise_config'])
 base=args.run/args.direction;phase=base/'phase1' if not o['phase1_from'] else Path(o['phase1_from'])/args.direction/'phase1'
 part=np.load(phase/'partition.npz');core=np.load(base/'coherence/representatives.npz')['moving_support'];free=np.load(phase/'state.npy',mmap_mode='r')[args.frame,:,:3];canonical=np.load(base/'rigid/canonical_estimate.npy');T=np.load(base/'rigid/poses.npy')[args.frame];rigid=canonical[core]@T[:3,:3].T+T[:3,3]
 frames=[data.observation(args.frame,ci) for ci in range(len(data.cameras))];raw=[samples(f,config) for f in frames];xyz=np.concatenate([r.xyz for r in raw]);rgb=np.concatenate([r.rgb for r in raw])
 import matplotlib
 matplotlib.use('Agg');import matplotlib.pyplot as plt
 fig=plt.figure(figsize=(17,11));ax=fig.add_subplot(2,2,1,projection='3d');ax.scatter(*xyz.T,c=rgb,s=1,alpha=.3);ax.scatter(*free[core].T,c='lime',s=1,label=f'Free proposed core ({int(core.sum())})');ax.set_title(f'Actual {len(xyz)} measured world points + free core');ax.legend()
 ax=fig.add_subplot(2,2,2,projection='3d');ax.scatter(*xyz.T,c=rgb,s=1,alpha=.3);ax.scatter(*rigid.T,c='red',s=1,label='Shared rigid core');ax.set_title('Shared rigid prediction (all selected points)');ax.legend()
 for ci in range(min(2,len(frames))):
  ax=fig.add_subplot(2,2,3+ci);ax.imshow(frames[ci].rgb);h,_=projection_numpy(rigid,frames[ci].camera);valid=np.isfinite(h).all(1)&(h[:,2]>0)&(h[:,0]>=0)&(h[:,0]<frames[ci].camera.width)&(h[:,1]>=0)&(h[:,1]<frames[ci].camera.height);ax.scatter(h[valid,0],h[valid,1],c='lime',s=1,alpha=.35);ax.set_xlim(0,frames[ci].camera.width);ax.set_ylim(frames[ci].camera.height,0);ax.set_title(f'Actual RGB camera {ci}; projected core incl. possible occlusion')
 fig.suptitle(f'{args.direction} / frame {args.frame} / t={data.times[args.frame]:.6f} s; proposal diagnostic, not accepted joint support');fig.tight_layout();output=args.run/f'reports/{args.direction}_core_{args.frame:06d}.png';output.parent.mkdir(exist_ok=True);fig.savefig(output,dpi=150);plt.close(fig);print(output)


if __name__=='__main__':main()
