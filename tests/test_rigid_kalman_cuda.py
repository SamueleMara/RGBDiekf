"""Gate F: SE(3) conventions by finite differences, the 12D rigid process Jacobian in both time directions, the prior-
chart Jacobian L, the point Jacobian B, the chart-reset Jacobian J_l (numerically, although production uses the
final-chart policy), and the rigid IEKF on ray-cast sequences (static, prismatic, revolute; forward and backward):
exact pairwise rigidity, final-chart covariance = local-chart Joseph."""
import unittest
import numpy as np
import torch
from scipy.linalg import expm, logm
from estimator import se3
from estimator.rigid_kalman_cuda import rigid_predict, RigidMeasurement, rigid_update, rigidity_check
from estimator.projective_camera import CameraRig
from estimator.projective_sensor import MeasuredFrame
from tests.projective_fixtures import noise, POLICY

D = torch.float64
g = np.random.default_rng(0)
rnd = lambda *s, sc=1.: torch.as_tensor(g.normal(scale=sc, size=s), dtype=D)


def fd_jacobian(f, x0, eps=1e-7):
    cols = []
    for i in range(len(x0)):
        d = torch.zeros_like(x0); d[i] = eps
        cols.append((f(x0+d)-f(x0-d))/(2*eps))
    return torch.stack(cols, -1)


class SE3Test(unittest.TestCase):
    def test_exp_log_hat_ad(self):
        for _ in range(50):
            xi = rnd(6, sc=.8)
            if float(xi[3:].norm()) > np.pi-.05: xi[3:] *= (np.pi-.05)/float(xi[3:].norm())    # principal branch of Log
            T = se3.exp(xi)
            np.testing.assert_allclose(T.numpy(), expm(se3.hat(xi).numpy()), atol=1e-12)
            np.testing.assert_allclose(se3.log(T).numpy(), xi.numpy(), atol=1e-10)
            eta = rnd(6)
            np.testing.assert_allclose(se3.hat(se3.Ad(T)@eta).numpy(), (T@se3.hat(eta)@se3.inv(T)).numpy(), atol=1e-12)
        for th in (1e-9, 1e-5, np.pi-1e-6, np.pi-1e-3):            # branch behaviour of Log
            w = torch.tensor([.3, -.5, .8], dtype=D); w = w/w.norm()*th
            xi = torch.cat([torch.tensor([.1, .2, -.3], dtype=D), w])
            np.testing.assert_allclose(se3.log(se3.exp(xi)).numpy(), xi.numpy(), atol=1e-7)

    def test_left_jacobian_and_inverse_by_finite_differences(self):
        for _ in range(20):
            xi = rnd(6, sc=.6)
            J = fd_jacobian(lambda d: se3.log(se3.exp(xi+d)@se3.inv(se3.exp(xi))), torch.zeros(6, dtype=D))
            np.testing.assert_allclose(se3.left_jacobian(xi).numpy(), J.numpy(), atol=1e-7)
            e = rnd(6, sc=.4)                                        # prior-chart Jacobian: e(Exp(d) T) = e + J_l(e)^-1 d
            Jinv = fd_jacobian(lambda d: se3.log(se3.exp(d)@se3.exp(e)), torch.zeros(6, dtype=D))
            np.testing.assert_allclose(se3.left_jacobian_inv(e).numpy(), Jinv.numpy(), atol=1e-7)

    def test_chart_reset_jacobian_is_left_jacobian(self):
        dtau = rnd(6, sc=.3)          # e_new = Log(Exp(e_old) Exp(-dtau)), differential at e_old = dtau
        J = fd_jacobian(lambda e: se3.log(se3.exp(e)@se3.exp(-dtau)), dtau.clone())
        np.testing.assert_allclose(J.numpy(), se3.left_jacobian(dtau).numpy(), atol=1e-7)

    def test_point_jacobian(self):
        x = rnd(5, 3)
        for i in range(5):
            J = fd_jacobian(lambda d: (se3.exp(d)[:3, :3]@x[i]+se3.exp(d)[:3, 3]), torch.zeros(6, dtype=D))
            np.testing.assert_allclose(se3.point_jacobian(x[i:i+1])[0].numpy(), J.numpy(), atol=1e-8)

    def test_process_jacobian_both_directions(self):
        Qc = torch.eye(6, dtype=D)*.01
        for dt in (1/30, -1/30, .4, -.4):
            T0, xi0 = se3.exp(rnd(6, sc=.5)), rnd(6, sc=.7)
            Tm, xim, _, F, Q = rigid_predict(T0, xi0, torch.eye(12, dtype=D), dt, Qc)
            def prop(d):
                T = se3.exp(d[:6])@T0; xi = xi0+d[6:]
                Tn = se3.exp(dt*xi)@T
                return torch.cat([se3.log(Tn@se3.inv(Tm)), xi-xim])
            np.testing.assert_allclose(F.numpy(), fd_jacobian(prop, torch.zeros(12, dtype=D)).numpy(), atol=1e-7)
            self.assertGreaterEqual(float(torch.linalg.eigvalsh(Q).min()), -1e-18)


def scene(motion, dev='cpu', frames=6, rate=.03, speed=.02):
    from tests.image_fixtures import two_plate_scene, look_at, catalog, source_parts
    ds = two_plate_scene(motion, frames=frames, cameras=4, rate=rate, speed=speed)
    ds.cameras = [look_at(c.name, -c.world_to_camera[:3, :3].T@c.world_to_camera[:3, 3], [0, 0, 0], 320, 240, 400.) for c in ds.cameras]
    return ds


def run_sequence(ds, direction, dev='cpu'):
    """Rigid IEKF over the sequence with the GT moving part as the core (TEST ONLY: inference never sees parts)."""
    from tests.image_fixtures import catalog, source_parts
    K = len(ds.times); k0 = 0 if direction == 'forward' else K-1
    xyz, rgb, _, source = catalog(ds, k0)
    parts = source_parts(ds, source)
    core, stat = np.flatnonzero(parts == 1), np.flatnonzero(parts == 0)
    t = lambda a: torch.as_tensor(a, dtype=D, device=dev)
    rig = CameraRig(ds.cameras, dev)
    T = torch.eye(4, dtype=D, device=dev); xi = torch.zeros(6, dtype=D, device=dev)
    P = torch.zeros((12, 12), dtype=D, device=dev); P[6:, 6:] = torch.eye(6, dtype=D, device=dev)
    Qc = torch.eye(6, dtype=D, device=dev)*.01
    order = range(K) if direction == 'forward' else range(K-1, -1, -1)
    order = list(order); out = {}
    for prev, k in zip(order, order[1:]):
        Tm, xim, Pm, _, _ = rigid_predict(T, xi, P, float(ds.times[k]-ds.times[prev]), Qc)
        meas = RigidMeasurement(rig, MeasuredFrame(ds, k, dev, D), noise(dev), POLICY, t(xyz[core]), t(rgb[core]), t(xyz[stat]),
                                torch.eye(3, dtype=D, device=dev).expand(len(stat), 3, 3)*1e-6)
        r = rigid_update(Tm, xim, Pm, meas, 10, 1e-10, trace=True)
        T, xi, P = r['T'], r['xi'], r['P']
        true = ds.poses(k)[1]@np.linalg.inv(ds.poses(k0)[1])
        out[k] = {'T': T.cpu().numpy(), 'true': true, 'trace': r['trace'], 'rigidity': rigidity_check(t(xyz[core]), T), 'P': P}
    return out


class RigidFilterTest(unittest.TestCase):
    def check(self, out, tol_t, tol_r):
        for k, o in out.items():
            dT = o['T']@np.linalg.inv(o['true'])
            self.assertLess(np.linalg.norm(dT[:3, 3]), tol_t, f'frame {k}')
            self.assertLess(np.linalg.norm(logm(dT[:3, :3]).real), tol_r, f'frame {k}')
            self.assertLess(o['rigidity']['max_pair_distance_error_m'], 1e-12)
            self.assertLess(o['trace']['final_chart_vs_local_joseph_max_abs'], 1e-10)
            self.assertGreater(float(torch.linalg.eigvalsh(o['P']).min()), 0)

    def test_prismatic_forward_and_backward(self):
        ds = scene('prismatic')
        for direction in ('forward', 'backward'):
            self.check(run_sequence(ds, direction), 3e-3, np.radians(.3))

    def test_revolute_forward_and_backward(self):
        ds = scene('revolute')
        for direction in ('forward', 'backward'):
            self.check(run_sequence(ds, direction), 3e-3, np.radians(.3))

    def test_static(self):
        self.check(run_sequence(scene('static'), 'forward'), 1e-3, np.radians(.1))


if __name__ == '__main__':
    unittest.main()
