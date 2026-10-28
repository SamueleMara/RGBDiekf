"""Calibrated camera rig on the GPU: world->camera transform, pinhole projection, its analytic Jacobian, back-projection.

Convention (contract section 1): T_cw is WORLD -> CAMERA, X_c = R_cw x_w + t_cw. scene.dataloader.camera_from_metadata
converts the stored camera-to-world pose ONCE at loading; both T_cw and T_wc are persisted in the manifest
(rig.manifest()). Pixel coordinates: u = column, v = row, the centre of pixel (row r, column c) is (c, r), so the
pixel with flat index p = r*W + c back-projects through (c, r) exactly as scene/endpoint.py does. Rectified images only
(scene/dataset.py refuses non-zero distortion that is not declared rectified).

    u = fx X/Z + cx,  v = fy Y/Z + cy,
    J_pi = [[fx/Z, 0, -fx X/Z^2], [0, fy/Z, -fy Y/Z^2]],  J_u = J_pi R_cw,  J_Z = e3^T R_cw.

Back-projection of a measured depth D along the PREDICTED ray b(u,v) = K^-1 [u, v, 1]^T:
    x_meas = R_cw^T (D b - t_cw),  and  x_meas - x_pred = R_cw^T b (D - Z_pred)   (rank-one ray-depth information).
"""
from dataclasses import dataclass
import hashlib
import numpy as np
import torch


@dataclass
class Projection:
    Xc: torch.Tensor     # [C,N,3] camera coordinates
    uv: torch.Tensor     # [C,N,2]
    z: torch.Tensor      # [C,N]   camera Z
    Ju: torch.Tensor     # [C,N,2,3] d(u,v)/dx_world
    Jz: torch.Tensor     # [C,3]   dZ/dx_world (= third row of R_cw)


class CameraRig:
    def __init__(self, cameras, device, dtype=torch.float64):
        T = np.stack([np.asarray(c.world_to_camera, float) for c in cameras])
        self.names = [c.name for c in cameras]
        self.width = [int(c.width) for c in cameras]; self.height = [int(c.height) for c in cameras]
        self.T_cw = T
        self.T_wc = np.linalg.inv(T)
        self.R = torch.as_tensor(T[:, :3, :3], dtype=dtype, device=device)
        self.t = torch.as_tensor(T[:, :3, 3], dtype=dtype, device=device)
        self.K = torch.as_tensor(np.stack([np.asarray(c.intrinsics, float) for c in cameras]), dtype=dtype, device=device)
        self.near = torch.as_tensor([float(c.near) for c in cameras], dtype=dtype, device=device)
        self.count = len(cameras)
        self.device, self.dtype = torch.device(device), dtype

    def to(self, dtype):
        out = object.__new__(CameraRig); out.__dict__.update(self.__dict__)
        out.R, out.t, out.K, out.near, out.dtype = self.R.to(dtype), self.t.to(dtype), self.K.to(dtype), self.near.to(dtype), dtype
        return out

    def calibration_hash(self):
        h = hashlib.sha256()
        for a in (self.T_cw, self.K.cpu().numpy(), np.asarray(self.width), np.asarray(self.height), self.near.cpu().numpy()):
            h.update(np.ascontiguousarray(a, dtype=np.float64).tobytes())
        return h.hexdigest()

    def manifest(self):
        return {'convention': 'T_cw world->camera (X_c = R_cw x_w + t_cw); T_wc = inv(T_cw) persisted; u=column, v=row, pixel centres at integers',
                'cameras': [{'name': n, 'width': w, 'height': h, 'intrinsics_fx_fy_cx_cy': k.tolist(), 'T_cw': tc.tolist(), 'T_wc': tw.tolist(),
                             'near': float(nr)} for n, w, h, k, tc, tw, nr in zip(self.names, self.width, self.height, self.K.cpu().numpy(),
                                                                                  self.T_cw, self.T_wc, self.near.cpu().numpy())],
                'calibration_hash': self.calibration_hash()}

    def project(self, x, cams=None, jacobian=True):
        """x [N,3] world -> Projection over all cameras (or the index tensor `cams`). jacobian=False skips J_u (visibility
        needs only (u, v, Z); the per-row Jacobian is recomputed by project_rows for the accepted rows)."""
        R, t, K = (self.R, self.t, self.K) if cams is None else (self.R[cams], self.t[cams], self.K[cams])
        x = x.to(self.dtype)
        Xc = torch.einsum('cij,nj->cni', R, x)+t[:, None, :]
        X, Y, Z = Xc.unbind(-1)
        safe = torch.where(Z.abs() > 0, Z, torch.ones_like(Z))
        fx, fy, cx, cy = (K[:, i, None] for i in range(4))
        uv = torch.stack((fx*X/safe+cx, fy*Y/safe+cy), -1)
        if not jacobian: return Projection(Xc, uv, Z, None, R[:, 2, :])
        Jpi = torch.zeros((*Z.shape, 2, 3), dtype=x.dtype, device=x.device)
        Jpi[..., 0, 0] = fx/safe; Jpi[..., 0, 2] = -fx*X/safe**2
        Jpi[..., 1, 1] = fy/safe; Jpi[..., 1, 2] = -fy*Y/safe**2
        Ju = torch.einsum('cnij,cjk->cnik', Jpi, R)
        return Projection(Xc, uv, Z, Ju, R[:, 2, :])

    def project_rows(self, x, cam):
        """Row-wise projection: x [M,3] each seen by camera cam [M] -> uv [M,2], z [M], Ju [M,2,3], Jz [M,3]."""
        R, t, K = self.R[cam], self.t[cam], self.K[cam]
        Xc = torch.einsum('mij,mj->mi', R, x.to(self.dtype))+t
        X, Y, Z = Xc.unbind(-1)
        safe = torch.where(Z.abs() > 0, Z, torch.ones_like(Z))
        fx, fy, cx, cy = K.unbind(-1)
        uv = torch.stack((fx*X/safe+cx, fy*Y/safe+cy), -1)
        Jpi = torch.zeros((len(x), 2, 3), dtype=Xc.dtype, device=Xc.device)
        Jpi[:, 0, 0] = fx/safe; Jpi[:, 0, 2] = -fx*X/safe**2
        Jpi[:, 1, 1] = fy/safe; Jpi[:, 1, 2] = -fy*Y/safe**2
        return uv, Z, (Jpi[:, :, :, None]*R[:, None, :, :]).sum(2), R[:, 2, :]

    def ray(self, cam, uv):
        """b(u,v) = K^-1 [u, v, 1]^T in camera coordinates, rows [M,3]."""
        K = self.K[cam]
        return torch.stack(((uv[:, 0]-K[:, 2])/K[:, 0], (uv[:, 1]-K[:, 3])/K[:, 1], torch.ones_like(uv[:, 0])), -1)

    def backproject(self, cam, uv, depth):
        """Measured depth D at (u,v) of camera cam -> world point R^T (D b - t) [M,3]."""
        b = self.ray(cam, uv)
        return torch.einsum('mji,mj->mi', self.R[cam], depth[:, None]*b-self.t[cam])
