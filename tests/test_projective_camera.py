"""Gate A (geometry): projection, analytic J_pi, back-projection and the rank-one ray-depth identity, against an
independent float64 NumPy oracle; calibration loading conventions (T_wc inverted once, metres only)."""
import json, tempfile, unittest
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from tests.projective_fixtures import camera, rot, DEVICES
from estimator.projective_camera import CameraRig


def oracle_project(x, R, t, K):
    Xc = x@R.T+t
    return np.stack((K[0]*Xc[:, 0]/Xc[:, 2]+K[2], K[1]*Xc[:, 1]/Xc[:, 2]+K[3]), 1), Xc[:, 2]


class ProjectionTest(unittest.TestCase):
    def setUp(self):
        g = np.random.default_rng(0)
        self.cams = [camera('a', rot([.3, -.5, .8], .7), np.array([.12, -.31, 2.2]), (510., 470., 51.7, 30.2)),
                     camera('b', rot([-.2, .9, .1], 1.9), np.array([-.4, .05, 1.7]), (600., 610., 40.1, 40.9))]
        self.x = g.normal(0, .25, (200, 3))

    def test_projection_and_jacobian_against_oracle(self):
        for dev in DEVICES:
            rig = CameraRig(self.cams, dev)
            p = rig.project(torch.as_tensor(self.x, device=dev))
            for c, cam in enumerate(self.cams):
                uv, z = oracle_project(self.x, cam.world_to_camera[:3, :3], cam.world_to_camera[:3, 3], cam.intrinsics)
                np.testing.assert_allclose(p.uv[c].cpu().numpy(), uv, rtol=0, atol=1e-9)
                np.testing.assert_allclose(p.z[c].cpu().numpy(), z, rtol=0, atol=1e-12)
                eps = 1e-6
                for a in range(3):
                    d = np.zeros(3); d[a] = eps
                    fd = (oracle_project(self.x+d, cam.world_to_camera[:3, :3], cam.world_to_camera[:3, 3], cam.intrinsics)[0] -
                          oracle_project(self.x-d, cam.world_to_camera[:3, :3], cam.world_to_camera[:3, 3], cam.intrinsics)[0])/(2*eps)
                    np.testing.assert_allclose(p.Ju[c, :, :, a].cpu().numpy(), fd, rtol=1e-6, atol=1e-6)
                np.testing.assert_allclose(p.Jz[c].cpu().numpy(), cam.world_to_camera[2, :3], atol=1e-15)

    def test_backprojection_roundtrip_and_rank_one_identity(self):
        rig = CameraRig(self.cams, 'cpu')
        x = torch.as_tensor(self.x)
        for c in range(2):
            cam = torch.full((len(x),), c)
            uv, z, _, _ = rig.project_rows(x, cam)
            np.testing.assert_allclose(rig.backproject(cam, uv, z).numpy(), self.x, atol=1e-12)
            D = z+torch.as_tensor(np.random.default_rng(c).normal(0, .05, len(x)))          # a different measured depth
            lhs = rig.backproject(cam, uv, D)-x
            rhs = torch.einsum('mji,mj->mi', rig.R[cam], rig.ray(cam, uv))*(D-z)[:, None]
            np.testing.assert_allclose(lhs.numpy(), rhs.numpy(), atol=1e-12)
            # rank one: the difference lies on the predicted ray, no tangential component
            ray_w = torch.einsum('mji,mj->mi', rig.R[cam], rig.ray(cam, uv))
            tang = lhs-(lhs*ray_w).sum(1, keepdim=True)/(ray_w*ray_w).sum(1, keepdim=True)*ray_w
            self.assertLess(float(tang.abs().max()), 1e-12)


class CalibrationLoadingTest(unittest.TestCase):
    def scene(self, depth, convention='opencv_optical_camera_to_world'):
        root = Path(tempfile.mkdtemp())
        c2w = np.eye(4); c2w[:3, :3] = rot([1, 2, 3], .4); c2w[:3, 3] = [.3, -.2, 1.5]
        (root/'metadata').mkdir()
        (root/'metadata/cameras.json').write_text(json.dumps({'cameras': [{'name': 'cam_000', 'width': 8, 'height': 6, 'fx': 10., 'fy': 11., 'cx': 3.5, 'cy': 2.5,
                                                                           'pose_convention': convention, 'pose_matrix_4x4': c2w.tolist()}]}))
        for d in ('rgb', 'depth_npy', 'mask'): (root/d/'cam_000').mkdir(parents=True)
        Image.fromarray(np.full((6, 8, 3), 100, np.uint8)).save(root/'rgb/cam_000/frame_000000.png')
        Image.fromarray(np.full((6, 8), 255, np.uint8)).save(root/'mask/cam_000/frame_000000.png')
        np.save(root/'depth_npy/cam_000/frame_000000.npy', depth)
        return root, c2w

    def test_camera_to_world_is_inverted_once_and_metres_are_required(self):
        from scene.dataset import Dataset
        root, c2w = self.scene(np.full((6, 8), 1.25, np.float32))
        d = Dataset(root, None, None, fps=30.)
        np.testing.assert_allclose(d.cameras[0].world_to_camera, np.linalg.inv(c2w), atol=1e-12)
        rig = CameraRig(d.cameras, 'cpu')
        np.testing.assert_allclose(rig.T_wc[0], c2w, atol=1e-12)
        f = d.observation(0, 0)
        self.assertTrue(np.all(f.depth == np.float32(1.25)))              # no implicit unit conversion
        root2, _ = self.scene(np.full((6, 8), 1250, np.uint16))         # millimetre integers are refused, never rescaled
        with self.assertRaises(ValueError):
            Dataset(root2, None, None, fps=30.).observation(0, 0)


if __name__ == '__main__':
    unittest.main()
