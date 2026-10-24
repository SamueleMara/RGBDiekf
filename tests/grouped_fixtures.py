"""Independent RGB-D sensor fixtures; truth is used only by test assertions."""
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from rgbd_common.projection_cuda import Camera
from rgbd_common.dataloader import RGBDFrame


class GroupScene:
    def __init__(self,frames=8,motion='static',occlusion=False,cameras=1):
        self.times=np.arange(frames)*.04;self.frame_ids=[40+2*k for k in range(frames)];self.root=Path('/tmp/grouped_rgbd_fixture');self.test_limits={}
        self.cameras=[]
        for ci in range(cameras):
            t=np.eye(4);t[0,3]=-.01*ci
            self.cameras.append(Camera(str(ci),96,72,np.array([100.,100.,47.5,35.5]),t,0,np.inf))
        u,v=np.meshgrid(np.arange(34,54),np.arange(28,38));self.points=np.c_[(u.ravel()-47.5)/100,(v.ravel()-35.5)/100,np.ones(u.size)]
        colors=np.random.default_rng(108).uniform(.05,.95,(len(self.points),3));self.images=[];self.truth=[]
        for k in range(frames):
            pose=np.eye(4)
            if motion=='translation':pose[0,3]=.01*k
            if motion=='variable_translation':pose[0,3]=.002*k+.001*k*k
            if motion=='rotation':
                pose[:3,:3]=Rotation.from_rotvec([0,.025*k,0]).as_matrix();axis=np.array([-.2,0,1.]);pose[:3,3]=axis-pose[:3,:3]@axis
            self.truth.append(pose.copy());xyz=self.points@pose[:3,:3].T+pose[:3,3]
            if motion=='boundary':
                xyz=self.points.copy();xyz[u.ravel()>=44,2]+=.018*k
            row=[]
            for ci,camera in enumerate(self.cameras):
                rgb=np.zeros((72,96,3));depth=np.zeros((72,96));mask=np.zeros((72,96),bool)
                q=xyz@camera.world_to_camera[:3,:3].T+camera.world_to_camera[:3,3]
                uv=np.floor(q[:,:2]/q[:,2,None]*100+[47.5,35.5]+.5).astype(int)
                for i,((x,y),point) in enumerate(zip(uv,q)):
                    if occlusion and 8<=k<35 and (i%20)>=10:continue
                    if point[2]>0 and 0<=x<96 and 0<=y<72 and (not mask[y,x] or depth[y,x]>point[2]):mask[y,x]=True;depth[y,x]=point[2];rgb[y,x]=colors[i]
                row.append(RGBDFrame(rgb,depth,mask,camera,k,ci))
            self.images.append(row)
    def observation(self,k,ci):return self.images[k][ci]
