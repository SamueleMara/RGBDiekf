from pathlib import Path
from types import SimpleNamespace
import numpy as np
from rgbd_common.projection_cuda import Camera
from rgbd_common.dataloader import RGBDFrame


class PointScene:
    def __init__(self,frames=5,cameras=2,motion='static',hidden=True):
        self.times=np.arange(frames)*.04;self.frame_ids=[100+3*k for k in range(frames)];self.root=Path('/tmp/pointwise_fixture');self.test_limits={}
        self.cameras=[]
        for ci in range(cameras):
            extrinsic=np.eye(4);extrinsic[0,3]=-.025*ci
            self.cameras.append(Camera(str(ci),64,48,np.array([80.,80.,31.5,23.5]),extrinsic,0,np.inf))
        self.images=[];self.truth=[];self.hidden_pixels={}
        for k in range(frames):
            first=np.array([-.08,0,1.])
            if motion=='translation':first[0]+=.008*k
            if motion=='rotation':first=np.array([-.08*np.cos(.12*k),-.08*np.sin(.12*k),1.])
            points=[first];colors=[[.8,.1,.2]]
            if hidden and k>=2:points.append(np.array([-.5,-.4,2.5]));colors.append([.1,.8,.2])
            row=[]
            for ci,cam in enumerate(self.cameras):
                rgb=np.zeros((48,64,3));depth=np.zeros((48,64));mask=np.zeros((48,64),bool)
                for pi,(point,color) in enumerate(zip(points,colors)):
                    q=cam.world_to_camera[:3,:3]@point+cam.world_to_camera[:3,3];uv=np.floor(q[:2]/q[2]*80+[31.5,23.5]+.5).astype(int);x,y=uv
                    if 0<=x<64 and 0<=y<48:
                        rgb[y,x]=color;depth[y,x]=q[2];mask[y,x]=True
                        if pi:self.hidden_pixels[k,ci]=y*64+x
                row.append(RGBDFrame(rgb,depth,mask,cam,k,ci,object_only_no_external_occluders=None))
            self.images.append(row);self.truth.append(first)
    def observation(self,k,c):return self.images[k][c]


def config():
    from pointwise_rgbd.config import load_config
    return load_config(tail_probability=.01)
