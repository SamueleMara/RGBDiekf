"""Shared helpers for the projective RGB-D tests: ray-cast sensor fields (tests/image_fixtures.py, NOT a renderer of the
estimator) and the declared policy/noise objects."""
import json
from pathlib import Path
import numpy as np
import torch
from scene.camera import Camera
from scene.config import load_config
from estimator.projective_camera import CameraRig
from estimator.projective_sensor import SensorNoise, MeasuredFrame

REPO = Path(__file__).resolve().parents[1]
POLICY = json.loads((REPO/'configs/projective_rgbd.json').read_text())
DEVICES = ['cpu']+(['cuda'] if torch.cuda.is_available() else [])


def noise(device='cpu', **over):
    n = SensorNoise.from_config(load_config(), POLICY, device)
    for k, v in over.items(): setattr(n, k, v)
    return n


def camera(name='c0', R=np.eye(3), t=np.zeros(3), K=(500., 520., 47.3, 36.1), W=96, H=72):
    g = np.eye(4); g[:3, :3] = R; g[:3, 3] = t
    return Camera(name, W, H, np.asarray(K, float), g, .01, 50.)


class FieldDataset:
    """Dataset-like wrapper around explicit per-camera images (analytic fields for unit tests)."""
    def __init__(self, cameras, frames):
        from scene.dataloader import RGBDFrame
        self.cameras = cameras; self.frames = frames; self.base = self
        self.times = np.arange(len(frames))/30.; self.frame_ids = list(range(len(frames)))
        self._RGBD = RGBDFrame

    def observation(self, k, ci):
        rgb, depth, mask = self.frames[k][ci]
        return self._RGBD(rgb, depth, mask, self.cameras[ci], k, ci, None, True)


def rot(axis, angle):
    from scipy.spatial.transform import Rotation
    return Rotation.from_rotvec(np.asarray(axis, float)/np.linalg.norm(axis)*angle).as_matrix()


def smooth_field(W=96, H=72, z0=2.0, slope=(0.004, -0.003), texture=True):
    """Smooth slanted-plane depth (metres) and smooth RGB texture; all foreground."""
    u, v = np.meshgrid(np.arange(W, dtype=float), np.arange(H, dtype=float))
    depth = z0+slope[0]*(u-W/2)+slope[1]*(v-H/2)
    if texture:
        rgb = np.stack((.5+.3*np.sin(u/7.)*np.cos(v/9.), .5+.25*np.sin((u+v)/11.), .45+.3*np.cos((u-.6*v)/6.)), -1)
    else:
        rgb = np.full((H, W, 3), .5)
    return rgb, depth, np.ones((H, W), bool)


def write_scene(ds, root):
    """Write a ray-cast SyntheticDataset to disk in the real dataset layout (metadata/cameras.json with an explicit
    world->camera optical pose, rgb/*.png, depth_npy/*.npy float32 metres, mask/*.png), so train.py runs on it."""
    import json
    from pathlib import Path
    from PIL import Image
    root = Path(root)
    cams = []
    for c in ds.cameras:
        fx, fy, cx, cy = map(float, c.intrinsics)
        cams.append({'name': c.name, 'width': c.width, 'height': c.height, 'fx': fx, 'fy': fy, 'cx': cx, 'cy': cy,
                     'pose_convention': 'opencv_optical_world_to_camera', 'pose_matrix_4x4': np.asarray(c.world_to_camera).tolist(), 'near': float(c.near)})
    (root/'metadata').mkdir(parents=True, exist_ok=True)
    (root/'metadata/cameras.json').write_text(json.dumps({'cameras': cams}))
    for k in range(len(ds.times)):
        for ci, c in enumerate(ds.cameras):
            f = ds.observation(k, ci)
            for d in ('rgb', 'depth_npy', 'mask'): (root/d/c.name).mkdir(parents=True, exist_ok=True)
            Image.fromarray(np.round(np.clip(f.rgb, 0, 1)*255).astype(np.uint8)).save(root/'rgb'/c.name/f'frame_{k:06d}.png')
            np.save(root/'depth_npy'/c.name/f'frame_{k:06d}.npy', np.where(f.mask, f.depth, 0.).astype(np.float32))
            Image.fromarray(np.where(f.mask, 255, 0).astype(np.uint8)).save(root/'mask'/c.name/f'frame_{k:06d}.png')
    return root


def hires(ds, W=320, H=240, f=400.):
    from tests.image_fixtures import look_at
    ds.cameras = [look_at(c.name, -c.world_to_camera[:3, :3].T@c.world_to_camera[:3, 3], [0, 0, 0], W, H, f) for c in ds.cameras]
    ds._cache = {}
    return ds
