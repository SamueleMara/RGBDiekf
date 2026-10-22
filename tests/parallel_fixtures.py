from pathlib import Path
from types import SimpleNamespace
import numpy as np
from rgbd_common.projection_cuda import Camera
from rgbd_common.dataloader import RGBDFrame
from rgbd_common.config import load_config


class Scene:
    def __init__(self,frames=4,cameras=2):
        self.times=np.arange(frames)*.04;self.frame_ids=list(range(frames))
        self.cameras=[Camera(str(c),8,8,np.array([10.,10.,3.5,3.5]),np.eye(4),0,np.inf) for c in range(cameras)]
        yy,xx=np.mgrid[:8,:8];rgb=np.stack([.2+.02*xx,.3+.01*yy,.4+.003*xx*yy],2)
        self.images=[[RGBDFrame(rgb.copy(),np.ones((8,8)),np.ones((8,8),bool),camera,k,c,object_only_no_external_occluders=True) for c,camera in enumerate(self.cameras)] for k in range(frames)]
        self.root=Path('/tmp/parallel_fixture');self.files=SimpleNamespace(exists=lambda _:False)
    def observation(self,k,c):return self.images[k][c]


def configuration():
    config=load_config()
    # Historical experimental objective, ONLY for component/integration
    # regression. This is not supplied theory or a production default.
    config.structure_prior=lambda model,states,times:(np.log(model.count+1)+np.log(model.count+2),np.log(len(states[0].poses))+np.log(len(states[0].poses)+1)+model.count*np.log(len(states[0].poses)))
    config.initializer_log_prior=lambda groups:-np.log(groups)-np.log(groups+1)
    return config


class VectorState:
    def __init__(self,value):self.value=np.asarray(value);self.dim=len(self.value);self.poses=np.eye(4)[None]
    def plus(self,delta):return VectorState(self.value+delta)
    def minus(self,base):return self.value-base.value
    def chart(self,delta):return np.eye(self.dim)


class LinearProblem:
    def __init__(self,h,r,z):self.h=h;self.r=r;self.z=z
    def evaluate(self,state):
        e=self.z-self.h@state.value;return float(e@np.linalg.solve(self.r,e)/2)


class DrawerScene:
    """Synthetic images only: a textured translating point occluder over points."""
    def __init__(self,frames=4,motion_per_frame=None):
        self.times=np.arange(frames)*.04;self.frame_ids=list(range(frames));self.root=Path('/tmp/drawer_fixture')
        self.cameras=[Camera('view',32,32,np.array([40.,40.,15.5,15.5]),np.eye(4),0,np.inf)]
        camera=self.cameras[0];yy,xx=np.mgrid[:32,:32];rng=np.random.default_rng(71)
        rear=np.c_[((xx.ravel()-15.5)/40),((yy.ravel()-15.5)/40),np.ones(1024)]
        rear_rgb=np.clip(np.array([.1,.3,.7])+rng.uniform(-.08,.08,(1024,3)),0,1)
        fy,fx=np.mgrid[8:24,9:23];front=np.c_[(fx.ravel()-15.5)*.7/40,(fy.ravel()-15.5)*.7/40,np.full(fx.size,.7)]
        front_rgb=np.clip(np.array([.8,.2,.1])+rng.uniform(-.08,.08,(len(front),3)),0,1)
        self.images=[]
        from rgbd_common.lie import exp
        motion=np.array([.04,0,0,0,0,0]) if motion_per_frame is None else np.asarray(motion_per_frame)
        for k in range(frames):
            transform=exp(k*motion);moving=front@transform[:3,:3].T+transform[:3,3]
            world=np.r_[rear,moving];colors=np.r_[rear_rgb,front_rgb]
            uv=np.floor(world[:,:2]/world[:,2,None]*40+15.5+.5).astype(int)
            inside=np.all((uv>=0)&(uv<32),axis=1);ids=np.flatnonzero(inside)
            ids=ids[np.argsort(world[ids,2],kind='stable')]
            depth=np.zeros((32,32));rgb=np.zeros((32,32,3))
            for i in ids:
                x,y=uv[i]
                if depth[y,x]==0:depth[y,x]=world[i,2];rgb[y,x]=colors[i]
            self.images.append([RGBDFrame(rgb,depth,np.ones((32,32),bool),camera,k,0)])
        self.files=SimpleNamespace(exists=lambda _:False)
    def observation(self,k,c):return self.images[k][c]
