"""Translation-first SE(3), left/world errors, right/body process noise."""
from dataclasses import dataclass

import numpy as np
from scipy.linalg import block_diag, expm
from scipy.spatial.transform import Rotation


def skew(x):
    a, b, c = x
    return np.array([[0., -c, b], [c, 0., -a], [-b, a, 0.]])


def ad(x):
    out = np.zeros((6, 6))
    out[:3, :3] = out[3:, 3:] = skew(x[3:])
    out[:3, 3:] = skew(x[:3])
    return out


def left_jacobian(x):
    # The upper right block is integral_0^1 exp(t ad_x) dt, even at x=0.
    a = np.zeros((12, 12))
    a[:6, :6], a[:6, 6:] = ad(x), np.eye(6)
    return expm(a)[:6, 6:]


def exp(x):
    x = np.asarray(x, dtype=float)
    w = skew(x[3:])
    a = np.linalg.norm(x[3:])
    if a < 1e-4:
        a2 = a*a
        b, c = .5-a2/24+a2*a2/720, 1/6-a2/120+a2*a2/5040
    else:
        b, c = (1-np.cos(a))/(a*a), (a-np.sin(a))/(a*a*a)
    t = np.eye(4)
    t[:3, :3] = Rotation.from_rotvec(x[3:]).as_matrix()
    t[:3, 3] = (np.eye(3)+b*w+c*w@w) @ x[:3]
    return t


def log(t):
    phi = Rotation.from_matrix(t[:3, :3]).as_rotvec()
    w = skew(phi)
    a = np.linalg.norm(phi)
    c = 1/12+a*a/720 if a < 1e-4 else (1-.5*a/np.tan(.5*a))/(a*a)
    return np.r_[(np.eye(3)-.5*w+c*w@w)@t[:3, 3], phi]


def inverse(t):
    out = np.eye(4)
    out[:3, :3] = t[:3, :3].T
    out[:3, 3] = -out[:3, :3]@t[:3, 3]
    return out


def adjoint(t):
    r = t[:3, :3]
    out = block_diag(r, r)
    out[:3, 3:] = skew(t[:3, 3])@r
    return out


@dataclass
class State:
    poses: np.ndarray
    twists: np.ndarray

    @property
    def dim(self):
        return 12*len(self.poses)

    def plus(self, d):
        d = np.asarray(d).reshape(-1, 12)
        return State(np.array([exp(v[:6])@t for t, v in zip(self.poses, d)]),
                     self.twists+d[:, 6:])

    def minus(self, base):
        return np.array([np.r_[log(t@inverse(u)), v-w] for t, u, v, w in
                         zip(self.poses, base.poses, self.twists, base.twists)]).ravel()

    def chart(self, delta):
        """Derivative from additive coordinates at base to left error at base+delta."""
        return block_diag(*[block_diag(left_jacobian(x[:6]), np.eye(6))
                            for x in np.asarray(delta).reshape(-1, 12)])


def source_map_covariance(world_point, source_pose, ry, ps, pys=None):
    """B2, including acquisition/pose cross terms when supplied."""
    r = source_pose[:3, :3].T
    jy = r
    js = r@np.c_[-np.eye(3), skew(world_point)]
    if pys is None:raise ValueError('Supply raw/source-pose cross covariance; explicit zero is an independence assumption')
    cross = np.asarray(pys)
    pp = jy@ry@jy.T+js@ps@js.T+jy@cross@js.T+js@cross.T@jy.T
    return pp, js@ps+jy@cross


def inverse_transport(world_point, source_pose, target_pose):
    p = inverse(source_pose)@np.r_[world_point, 1.]
    return (target_pose@p)[:3]

