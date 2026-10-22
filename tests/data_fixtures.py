import json
import numpy as np
from PIL import Image
from rgbd_common.lie import exp


def make_dataset(root, frames=2, cameras=1):
    (root/"metadata").mkdir()
    specs = []
    for c in range(cameras):
        name = f"cam_{c:03d}"
        specs.append(dict(name=name, width=2, height=2, fx=3., fy=3., cx=1., cy=1.,
                          near=.01, far=10., pose_matrix_4x4=exp(
                              np.array([.1+c*.01, -.2, .3, .1, -.15, .08])).tolist()))
        (root/"rgb"/name).mkdir(parents=True)
        (root/"depth_npy"/name).mkdir(parents=True)
        (root/"masks"/name).mkdir(parents=True)
        for k in range(frames):
            Image.fromarray(np.full((2, 2, 3), 100+k, dtype=np.uint8)).save(root/"rgb"/name/f"frame_{k:06d}.png")
            Image.fromarray(np.full((2,2),255,dtype=np.uint8)).save(root/"masks"/name/f"frame_{k:06d}.png")
            depth = np.zeros((2, 2), dtype=np.float32)
            depth[0, 0] = 2.+k*.01
            np.save(root/"depth_npy"/name/f"frame_{k:06d}.npy", depth)
    (root/"metadata/cameras.json").write_text(json.dumps({"cameras": specs}))
    return specs

