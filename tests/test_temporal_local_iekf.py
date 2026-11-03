"""Point-local IEKF against an independent NumPy model of the SAME local Gaussian (spec tests 2,3,5-8,11-13).

The witness re-implements the spec equation f = sum_a w_a (y_a - eta_a) + e4 (Z_t - Z_ref) + eta_cur with NumPy
projection, finite-difference Jacobians and QR / null-space least squares. Only the discrete row domain (owner pixel,
reference cell, raw keys) is taken from the production visibility builder, as the environment of the test.
"""
import unittest
from types import SimpleNamespace
import numpy as np
import torch
from tests.test_temporal_warp import rig, frame
from estimator.projective_sensor import SensorNoise
from estimator.temporal_reference import ReferenceTable
from estimator.temporal_zbuffer import first_hit_pixel_owner
from estimator.temporal_rgbd_terms import build_temporal_rows
from estimator.temporal_simplified_iekf import update_points_temporal_simplified
from estimator.temporal_local_covariance import STATE, CHART, KEY, Codec, Layout, LocalGaussian, SourceChart, LocalCovarianceFailure
from estimator.temporal_local_iekf import LocalFrameUpdate

DEVICES = ['cpu']+(['cuda'] if torch.cuda.is_available() else [])
POLICY = {'same_surface_n_sigma': 3., 'same_surface_max_incidence_deg': 80.}
RGB_VAR, DEPTH_VAR = 1e-4, 1e-6
LOCAL_WITNESSES = []


def noise(device):
    return SensorNoise(torch.full((3,), RGB_VAR, dtype=torch.float64, device=device), DEPTH_VAR, 0.)


def sensor_var(v):
    return np.r_[np.full(3, RGB_VAR), DEPTH_VAR]


def correlated_cov(scales, rng, rho=.6):
    A = rng.standard_normal((len(scales), len(scales)))
    S = A@A.T+len(scales)*np.eye(len(scales))
    d = np.sqrt(np.diag(S))
    C = (1-rho)*np.eye(len(scales))+rho*S/np.outer(d, d)
    return C*np.outer(scales, scales)


class Problem:
    """N points, each with prior [s_t | reference block | retained keys], current frame 1, reference frame 0."""
    def __init__(self, device, points, warp, anchor=False, textureless=False, shift=.03, seed=0):
        self.device, self.anchor = device, anchor
        self.r = rig(device)
        self.f0, self.f1 = frame(device), frame(device)
        self.f1.k = 1
        if textureless:
            for f in (self.f0, self.f1):
                f.rgb[:] = .5
                f.depth[:] = 2.
        else:
            self.f1.rgb[..., 0] -= .02*2*shift
            self.f1.rgb[..., 1] -= .03*2*shift
        self.frames = {0: self.f0, 1: self.f1}
        N = len(points)
        self.codec = Codec(2, 100 if not anchor else 2, 64)
        rng = np.random.default_rng(seed)
        K, R, t = self.r.K.cpu().numpy(), self.r.R.cpu().numpy(), self.r.t.cpu().numpy()
        self.np_rig = (K, R, t)
        blocks, means, covs = [], [], []
        anchor_frame = 0 if anchor else 99
        src_cam, src_pix, base, D0 = [], [], [], []
        for i, (xt, xr, retained) in enumerate(points):
            if anchor:
                cam, pix = 0, int(np.rint(K[0, 3]+K[0, 1]*xr[1]/xr[2]))*8+int(np.rint(K[0, 2]+K[0, 0]*xr[0]/xr[2]))
                uv0 = np.array([pix % 8, pix//8], float)
                d0 = float(self.f0.depth[0, pix//8, pix % 8])
                b = R[0].T@(d0*np.array([(uv0[0]-K[0, 2])/K[0, 0], (uv0[1]-K[0, 3])/K[0, 1], 1.])-t[0])
                src_cam.append(cam), src_pix.append(pix), base.append(b), D0.append(d0)
                blocks += [(i, STATE, 0, 1, 0), (i, CHART, 0, 0, 0), (i, KEY, 0, 0, pix)]
                m = np.r_[xt, np.zeros(5), np.zeros(4)]
                scales = np.r_[[1e-2]*3, [1.]*3, [.3, .3], [1.]*3, [1e-2]*3, 1e-3]
            else:
                blocks += [(i, STATE, 0, 1, 0), (i, STATE, 0, 0, 0)]+[(i, KEY, 0, 0, p) for p in retained]
                m = np.r_[xt, xr]
                scales = np.r_[[1e-2]*3, [1.]*3, [1e-2]*3, [1.]*3]
                for _ in retained:
                    m = np.r_[m, rng.normal(0, [3e-3, 3e-3, 3e-3, 3e-4])]
                    scales = np.r_[scales, [1e-2]*3, 1e-3]
            means.append(m)
            covs.append(correlated_cov(scales, rng))
        t_ = lambda j: torch.tensor([b[j] for b in blocks], device=device)
        self.layout = Layout(N, t_(0), t_(1), t_(2), t_(3), t_(4))
        self.prior = LocalGaussian.empty(self.layout)
        for i in range(N):
            self.prior.scatter(torch.tensor([i], device=device), torch.tensor(means[i][None], device=device), torch.tensor(covs[i][None], device=device))
        self.means, self.covs = means, covs
        if anchor:
            self.chart = SourceChart(self.r, 0, torch.tensor(src_cam, device=device), torch.tensor(src_pix, device=device), 8,
                                     torch.tensor(np.array(base), device=device), torch.tensor(D0, dtype=torch.float64, device=device))
        else:
            self.chart = SourceChart(self.r, anchor_frame, torch.zeros(N, dtype=torch.long, device=device), torch.zeros(N, dtype=torch.long, device=device),
                                     8, torch.zeros((N, 3), dtype=torch.float64, device=device), torch.ones(N, dtype=torch.float64, device=device))
        self.refs = ReferenceTable(2, N, device, torch.float64)
        self.refs.frame[:] = 0
        self.refs.ordinal[:] = 0
        self.noise = noise(device)
        self.warp = torch.as_tensor(warp, dtype=torch.float64, device=device)
        self.expired = torch.full((2, N), -1, dtype=torch.long, device=device)
        self.frame_ordinal = torch.tensor([0, 1]+[-1]*98, device=device)[:self.codec.K]
        self.build_calls = 0

    def build(self, states):
        self.build_calls += 1
        own = first_hit_pixel_owner(states[:, :3], self.r, self.f1, dense=False)
        rows, reason, life = build_temporal_rows(states, own, self.f1, self.refs, self.frames, self.r, self.noise, POLICY, self.warp, 1, 3)
        return rows, own, reason, life

    def update(self, build=None):
        return LocalFrameUpdate(self.prior, self.chart, self.codec, self.refs, build or self.build, self.f1, 1, 1, 3, self.r, self.noise,
                                self.warp, 10**8, self.expired, self.frame_ordinal)


class Witness:
    """Independent NumPy local model of ONE point at a fixed discrete domain."""
    def __init__(self, problem, update, lr, point, layout_blocks, prior_mean, prior_cov):
        self.K, self.R, self.t = problem.np_rig
        self.blocks = layout_blocks
        self.col = {}
        for blk in layout_blocks:
            key = (blk['kind'], blk['camera'], blk['frame'], blk['pixel'])
            self.col[key] = slice(*blk['columns'])
        self.n = layout_blocks[-1]['columns'][1]
        self.rows = []
        ix = torch.where(lr.rows.point == point)[0].tolist()
        for a in ix:
            r = lr.rows
            self.rows.append({'camera': int(r.camera[a]), 'pixel': int(r.pixel[a]), 'rf': int(r.reference_frame[a]),
                              'cell': r.cell[a].cpu().numpy().astype(float), 'refpix': r.reference_pixels[a].tolist(),
                              'raw_ref': r.values[a].cpu().numpy(), 'y': r.y[a].cpu().numpy()})
        self.anchor = problem.anchor
        if problem.anchor:
            ch = problem.chart
            self.src = (int(ch.camera[point]), int(ch.pixel[point]), float(ch.depth[point]))
        self.mu, self.Sigma = prior_mean, prior_cov

    def project(self, c, x):
        X = self.R[c]@x+self.t[c]
        fx, fy, cx, cy = self.K[c]
        return np.array([fx*X[0]/X[2]+cx, fy*X[1]/X[2]+cy]), X[2]

    def reference_state(self, a, rf):
        if self.anchor and rf == 0:
            th = a[self.col[('source_chart', None, 0, None)]]
            c, pix, D0 = self.src
            eta = a[self.col[('raw_sensor_error', c, 0, pix)]][3]
            fx, fy, cx, cy = self.K[c]
            u, v = pix % 8+th[0], pix//8+th[1]
            x = self.R[c].T@((D0-eta)*np.array([(u-cx)/fx, (v-cy)/fy, 1.])-self.t[c])
            return np.r_[x, th[2:]]
        return a[self.col[('state', None, rf, None)]]

    def f(self, a):
        out = []
        st = a[self.col[('state', None, 1, None)]]
        for row in self.rows:
            c = row['camera']
            sr = self.reference_state(a, row['rf'])
            ut, zt = self.project(c, st[:3])
            ur, zr = self.project(c, sr[:3])
            q = np.array([row['pixel'] % 8, row['pixel']//8], float)-ut+ur
            u, v = q-row['cell']
            w = np.array([(1-u)*(1-v), u*(1-v), (1-u)*v, u*v])
            clean = np.stack([row['raw_ref'][j]-a[self.col[('raw_sensor_error', c, row['rf'], p)]] for j, p in enumerate(row['refpix'])])
            h = w@clean
            h[3] += zt-zr
            out.append(h+a[self.col[('raw_sensor_error', c, 1, row['pixel'])]])
        return np.concatenate(out)

    def y(self):
        return np.concatenate([row['y'] for row in self.rows])

    def jacobian(self, a, h=1e-4):
        """Five-point central differences (truncation O(h^4))."""
        cols = []
        for j in range(self.n):
            e = np.zeros(self.n)
            e[j] = h
            cols.append((-self.f(a+2*e)+8*self.f(a+e)-8*self.f(a-e)+self.f(a-2*e))/(12*h))
        return np.stack(cols, 1)

    def qr_update(self, a0, Rw):
        """Linearized MAP at a0 with FIXED prior, by QR (Rw SPD) or null-space QR (Rw = 0), and its covariance."""
        H = self.jacobian(a0)
        r = self.y()-self.f(a0)+H@(a0-self.mu)
        Lp = np.linalg.cholesky(self.Sigma)
        if np.any(Rw):
            Lr = np.linalg.cholesky(blockdiag(*[Rw]*len(self.rows)))
            A = np.vstack((np.linalg.inv(Lp), np.linalg.solve(Lr, H)))
            b = np.r_[np.zeros(self.n), np.linalg.solve(Lr, r)]
            Q, Rq = np.linalg.qr(A)
            d = np.linalg.solve(Rq, Q.T@b)
            Ri = np.linalg.inv(Rq)
            return self.mu+d, Ri@Ri.T
        B = H@Lp
        Q, Rq = np.linalg.qr(B.T)
        u = Q@np.linalg.solve(Rq.T, r)
        P = np.eye(self.n)-Q@Q.T
        return self.mu+Lp@u, Lp@P@Lp.T


def blockdiag(*a):
    n = sum(len(x) for x in a)
    out = np.zeros((n, n))
    i = 0
    for x in a:
        out[i:i+len(x), i:i+len(x)] = x
        i += len(x)
    return out


def witness_for(problem, update, lr, point):
    layout, origin, _ = update._work_layout()
    blocks = layout.describe(point)
    n = blocks[-1]['columns'][1]
    mu = np.zeros(n)
    Sigma = np.zeros((n, n))
    pm, pc = problem.means[point], problem.covs[point]
    mu[:len(pm)] = pm
    Sigma[:len(pm), :len(pm)] = pc
    for blk in blocks[len(problem.layout.describe(point)):]:
        a, b = blk['columns']
        Sigma[a:b, a:b] = np.diag(sensor_var(None))
    return Witness(problem, update, lr, point, blocks, mu, Sigma)


def production_vector(update, point, name):
    layout, origin, _ = update._work_layout()
    out = []
    nprior = len(update.prior.layout)
    for j in range(int(layout.boff[point]), int(layout.boff[point+1])):
        o = int(origin[j])
        if o < nprior:
            a = int(update.prior.moff[point]+update.prior.layout.start[o])
            out.append(update.P[name][a:a+int(layout.size[j])].cpu().numpy())
        else:
            out.append(update.ex.v[name][o-nprior].cpu().numpy())
    return np.concatenate(out)


NONIDEAL = np.diag([2e-5, 3e-5, 4e-5, 2e-7])+np.full((4, 4), 1e-8)-np.diag([1e-8]*4)+np.diag([1e-8]*4)


def one_point(device, warp, anchor=False, textureless=False):
    xt = np.array([.013, -.021, 2., .1, 0, 0])
    xr = np.array([.02, -.01, 2., .1, 0, 0])
    # textureless: no retained key (a nonzero posterior eta mean would itself texture the CLEAN reference image)
    return Problem(device, [(xt, xr, [28])] if not (anchor or textureless) else [(xt, xr, [])], warp, anchor=anchor, textureless=textureless)


class JacobianTests(unittest.TestCase):
    def test_two_time_and_raw_error_jacobians(self):
        for device in DEVICES:
            for anchor in (False, True):
                with self.subTest(device=device, anchor=anchor):
                    pb = one_point(device, NONIDEAL, anchor)
                    up = pb.update()
                    lr = up.linearize('it')
                    self.assertEqual(len(lr), 2)
                    layout, origin, index = up._work_layout()
                    pts = torch.tensor([0], device=device)
                    mu, Sigma, a, loc, _ = up._dense(pts, 'it', layout, origin)
                    H, r, Rw, valid, _ = up._system(pts, lr, loc, index, mu.shape[1])
                    wit = witness_for(pb, up, lr, 0)
                    a0 = production_vector(up, 0, 'it')
                    Hfd = wit.jacobian(a0)
                    Hp = H[0][valid[0]].cpu().numpy()
                    order = np.argsort([row['camera'] for row in wit.rows])
                    Hfd = np.concatenate([Hfd[4*j:4*j+4] for j in order])
                    np.testing.assert_allclose(Hp, Hfd, rtol=0, atol=2e-7*max(1, np.abs(Hfd).max()))
                    np.testing.assert_array_equal(Hp[:, 3:6], 0)              # current velocity: no direct column
                    if not anchor:
                        np.testing.assert_array_equal(Hp[:, 9:12], 0)         # reference velocity: no direct column
                    res = np.concatenate([wit.y()[4*j:4*j+4]-wit.f(a0)[4*j:4*j+4] for j in order])
                    np.testing.assert_allclose(r[0][valid[0]].cpu().numpy(), res, rtol=0, atol=1e-13)
                    LOCAL_WITNESSES.append({'test': 'jacobian', 'device': device, 'anchor_chart': anchor,
                                            'max_abs_H_minus_FD': float(np.abs(Hp-Hfd).max())})

    def test_one_key_one_variable_and_shared_innovation_covariance(self):
        """The source raw-depth error enters the chart AND a reference neighbour: one variable, columns summed."""
        pb = one_point('cpu', NONIDEAL, anchor=True)
        up = pb.update()
        lr = up.linearize('it')
        layout, origin, index = up._work_layout()
        src = (0, 0, int(pb.chart.pixel[0]))
        keys = [(int(layout.camera[j]), int(layout.frame[j]), int(layout.pixel[j])) for j in range(len(layout)) if int(layout.kind[j]) == KEY]
        self.assertEqual(keys.count(src), 1)
        row0 = torch.where(lr.rows.camera == 0)[0]
        self.assertIn(src[2], lr.rows.reference_pixels[row0[0]].tolist())
        self.assertGreater(lr.reused_source_keys, 0)
        pts = torch.tensor([0])
        mu, Sigma, a, loc, _ = up._dense(pts, 'it', layout, origin)
        H, r, Rw, valid, _ = up._system(pts, lr, loc, index, mu.shape[1])
        S = (H@Sigma@H.transpose(1, 2))[0][valid[0]][:, valid[0]].numpy()
        wit = witness_for(pb, up, lr, 0)
        Hfd = wit.jacobian(production_vector(up, 0, 'it'))
        order = np.argsort([row['camera'] for row in wit.rows])
        Hfd = np.concatenate([Hfd[4*j:4*j+4] for j in order])
        expected = Hfd@wit.Sigma@Hfd.T
        np.testing.assert_allclose(S, expected, rtol=0, atol=1e-9*np.abs(expected).max())
        self.assertGreater(np.abs(S[:4, 4:]).max(), 1e-8)                    # cross-camera innovation covariance
        LOCAL_WITNESSES.append({'test': 'shared_key', 'cross_camera_innovation_max_abs': float(np.abs(S[:4, 4:]).max())})


class UpdateTests(unittest.TestCase):
    def check_step(self, device, warp, anchor=False, textureless=False):
        pb = one_point(device, warp, anchor, textureless)
        up = pb.update()
        lr = up.linearize('it')
        observed = torch.zeros(1, dtype=torch.bool, device=device)
        observed[0] = True
        up.proposal(lr, observed)
        wit = witness_for(pb, up, lr, 0)
        a1, _ = wit.qr_update(wit.mu, warp)
        np.testing.assert_allclose(production_vector(up, 0, 'prop'), a1, rtol=0, atol=1e-10*max(1, np.abs(a1).max()))
        return pb, up, wit

    def test_stacked_multiview_step_equals_dense_qr(self):
        for device in DEVICES:
            for warp in (NONIDEAL, np.zeros((4, 4))):
                for anchor in (False, True):
                    with self.subTest(device=device, ideal=not np.any(warp), anchor=anchor):
                        self.check_step(device, warp, anchor)

    def test_posterior_joseph_equals_qr_and_algebraic_forms(self):
        for device in DEVICES:
            for warp in (NONIDEAL, np.zeros((4, 4))):
                with self.subTest(device=device, ideal=not np.any(warp)):
                    pb = one_point(device, warp)
                    up = pb.update()
                    solver = up.solve(10, 1e-12)
                    self.assertTrue(bool(up.assimilated[0]))
                    new, s, P, post = up.posterior(pb.refs, [0])
                    tr = post['traces'][0]
                    wit = Witness(pb, up, up.final, 0, tr['posterior_layout'], *self.prior_on(pb, tr['posterior_layout']))
                    a_star = np.asarray(tr['mean_plus'])
                    H = wit.jacobian(a_star)
                    Lp = np.linalg.cholesky(wit.Sigma)
                    if np.any(warp):
                        Rw = blockdiag(*[warp]*len(wit.rows))
                        info = np.linalg.inv(np.linalg.inv(wit.Sigma)+H.T@np.linalg.solve(Rw, H))
                        Kg = wit.Sigma@H.T@np.linalg.inv(H@wit.Sigma@H.T+Rw)
                        np.testing.assert_allclose(np.asarray(tr['Sigma_plus']), info, rtol=0, atol=1e-9*np.abs(info).max())
                        np.testing.assert_allclose(np.asarray(tr['Sigma_plus']), (np.eye(wit.n)-Kg@H)@wit.Sigma, rtol=0, atol=1e-9*np.abs(info).max())
                    else:
                        Q, _ = np.linalg.qr((H@Lp).T)
                        proj = Lp@(np.eye(wit.n)-Q@Q.T)@Lp.T
                        np.testing.assert_allclose(np.asarray(tr['Sigma_plus']), proj, rtol=0, atol=1e-9*np.abs(proj).max())
                        Sx = H@wit.Sigma@H.T
                        self.assertGreater(np.linalg.eigvalsh(Sx).min(), .99*DEPTH_VAR)   # SPD from current sensor errors alone
                    self.assertLess(post['joseph_vs_algebraic_max_correlation_units'], 1e-9)
                    # velocity: no direct column, nonzero posterior change through correlation
                    self.assertGreater(abs(float(s[0, 3]-pb.means[0][3])), 1e-6)
                    LOCAL_WITNESSES.append({'test': 'posterior', 'device': device, 'ideal': not np.any(warp),
                                            'iterations': len(solver['history']), 'velocity_change_m_s': float(s[0, 3]-pb.means[0][3])})

    @staticmethod
    def prior_on(pb, blocks):
        n = blocks[-1]['columns'][1]
        mu, Sigma = np.zeros(n), np.zeros((n, n))
        m = len(pb.means[0])
        mu[:m], Sigma[:m, :m] = pb.means[0], pb.covs[0]
        for blk in blocks:
            if blk['columns'][0] >= m:
                a, b = blk['columns']
                Sigma[a:b, a:b] = np.diag(sensor_var(None))
        return mu, Sigma

    def test_sequential_equals_stacked_in_linear_limit(self):
        pb = one_point('cpu', NONIDEAL)
        up = pb.update()
        up.solve(10, 1e-12)
        new, s, P, post = up.posterior(pb.refs, [0])
        tr = post['traces'][0]
        wit = Witness(pb, up, up.final, 0, tr['posterior_layout'], *self.prior_on(pb, tr['posterior_layout']))
        H = wit.jacobian(np.asarray(tr['mean_plus']))
        Sigma = wit.Sigma.copy()
        for j in range(len(wit.rows)):
            Hj = H[4*j:4*j+4]
            Kj = Sigma@Hj.T@np.linalg.inv(Hj@Sigma@Hj.T+NONIDEAL)
            Sigma = (np.eye(wit.n)-Kj@Hj)@Sigma@(np.eye(wit.n)-Kj@Hj).T+Kj@NONIDEAL@Kj.T
        np.testing.assert_allclose(np.asarray(tr['Sigma_plus']), Sigma, rtol=0, atol=1e-9*np.abs(Sigma).max())

    def test_textureless_flat_depth_needs_no_noise_floor(self):
        for device in DEVICES:
            pb, up, wit = self.check_step(device, np.zeros((4, 4)), textureless=True)
            H = wit.jacobian(wit.mu)
            np.testing.assert_allclose(H[:3, :3], 0, atol=1e-9)                # no RGB texture: no RGB geometry information
            S = H@wit.Sigma@H.T
            self.assertGreaterEqual(np.linalg.eigvalsh(S).min(), DEPTH_VAR*(1-1e-9))

    def test_invalid_covariance_fails_without_damping(self):
        pb = one_point('cpu', np.zeros((4, 4)))
        pb.prior.cov[:] = float('nan')
        up = pb.update()
        lr = up.linearize('it')
        with self.assertRaises(LocalCovarianceFailure):
            up.proposal(lr, torch.tensor([True]))


class AcceptanceTests(unittest.TestCase):
    def test_fixed_prior_and_rejected_iterates_add_no_information(self):
        pb = one_point('cpu', NONIDEAL)
        mean0, cov0 = pb.prior.mean.clone(), pb.prior.cov.clone()
        up = pb.update()
        up.solve(10, 1e-12)
        torch.testing.assert_close(pb.prior.mean, mean0, rtol=0, atol=0)
        torch.testing.assert_close(pb.prior.cov, cov0, rtol=0, atol=0)
        self.assertIs(up.mu, pb.prior.mean)
        # every moved trial loses its rows -> the common domain rejects it -> prediction only
        pb2 = one_point('cpu', NONIDEAL)
        base = pb2.prior.mean[:6].clone()

        def hostile(states):
            rows, own, reason, life = pb2.build(states)
            if not torch.equal(states[0], base):
                keep = torch.zeros(len(rows.point), dtype=torch.bool)
                for name in rows.__dataclass_fields__:
                    setattr(rows, name, getattr(rows, name)[keep])
            return rows, own, reason, life
        up2 = pb2.update(hostile)
        solver = up2.solve(10, 1e-12)
        self.assertFalse(bool(up2.assimilated[0]))
        self.assertEqual(solver['history'][0]['reason'], 'COMMON_DOMAIN_OR_MASK_OR_DESCENT_REJECTED')
        new, s, P, post = up2.posterior(pb2.refs, [0])
        np.testing.assert_array_equal(np.asarray(post['traces'][0]['Sigma_plus']), pb2.covs[0])
        np.testing.assert_array_equal(s[0].numpy(), pb2.means[0][:6])
        self.assertEqual(solver['nonconverged_points'], 1)

    def test_no_row_point_keeps_exact_prediction(self):
        xt = np.array([.013, -.021, 2., .1, 0, 0])
        far = np.array([50., 50., 2., .3, .2, .1])
        pb = Problem('cpu', [(xt, xt+[.007, .011, 0, 0, 0, 0], [28]), (far, far, [])], NONIDEAL)
        up = pb.update()
        up.solve(10, 1e-12)
        new, s, P, post = up.posterior(pb.refs, [1])
        self.assertFalse(bool(up.assimilated[1]))
        np.testing.assert_array_equal(s[1].numpy(), far)
        np.testing.assert_array_equal(np.asarray(post['traces'][1]['Sigma_plus']), pb.covs[1])


class CrossPointTests(unittest.TestCase):
    def test_shared_reference_pixel_is_an_independent_local_copy(self):
        """Global dense oracle: P_ij != 0. Local mode: independent per-point copies, P_ij = 0 by declared assumption."""
        xa = np.array([.013, -.021, 2., .1, 0, 0])
        xb = np.array([.513, -.021, 2., .1, 0, 0])
        pb = Problem('cpu', [(xa, xa+[.007, .011, 0, 0, 0, 0], []), (xb, xb+[.007, .011, 0, 0, 0, 0], [])], NONIDEAL, seed=4)
        up = pb.update()
        lr = up.linearize('it')
        ra = set(map(tuple, [(int(c), int(p)) for c, pp in zip(lr.rows.camera.tolist(), lr.rows.reference_pixels.tolist()) for p in pp]))
        shared = [(c, p) for c, p in ra if sum((int(c2) == c and p in pp) for c2, pp in zip(lr.rows.camera.tolist(), lr.rows.reference_pixels.tolist())) > 1]
        self.assertTrue(shared, 'fixture must share a raw reference pixel between the two points')
        observed = torch.ones(2, dtype=torch.bool)
        up.proposal(lr, observed)
        wits = [witness_for(pb, up, lr, i) for i in range(2)]
        for i, w in enumerate(wits):
            a1, P1 = w.qr_update(w.mu, NONIDEAL)
            np.testing.assert_allclose(production_vector(up, i, 'prop'), a1, rtol=0, atol=1e-10*max(1, np.abs(a1).max()))
        # global oracle: ONE variable per shared raw key across both points
        names = []
        for i, w in enumerate(wits):
            for blk in w.blocks:
                key = (blk['kind'], blk['camera'], blk['frame'], blk['pixel'])
                shared_key = blk['kind'] == 'raw_sensor_error' and (blk['camera'], blk['pixel']) in shared and blk['frame'] == 0
                names.append(('shared',)+key if shared_key else (i,)+key)
        unique = list(dict.fromkeys(names))
        width = {}
        per_point = []
        for i, w in enumerate(wits):
            block_names = names[sum(len(x.blocks) for x in wits[:i]):][:len(w.blocks)]
            per_point.append(block_names)
            for nm, blk in zip(block_names, w.blocks):
                width[nm] = blk['columns'][1]-blk['columns'][0]
        offs = dict(zip(unique, np.cumsum([0]+[width[nm] for nm in unique])[:-1]))
        flat = [np.concatenate([np.arange(offs[nm], offs[nm]+width[nm]) for nm in block_names]) for block_names in per_point]
        n = sum(width.values())
        mu, Sigma = np.zeros(n), np.zeros((n, n))
        for w, f in zip(wits, flat):
            # shared keys are fresh in both points: independent prior N(0,R), identical in both copies
            mu[f] = w.mu
            Sigma[np.ix_(f, f)] = w.Sigma
        Hs = [w.jacobian(w.mu) for w in wits]
        Hg = np.zeros((sum(len(h) for h in Hs), n))
        r = []
        row = 0
        for w, f, h in zip(wits, flat, Hs):
            Hg[row:row+len(h), f] = h
            r.append(w.y()-w.f(w.mu))
            row += len(h)
        Rw = blockdiag(*[NONIDEAL]*(row//4))
        Pg = np.linalg.inv(np.linalg.inv(Sigma)+Hg.T@np.linalg.solve(Rw, Hg))
        cross = Pg[np.ix_(flat[0][:6], flat[1][:6])]
        local = [w.qr_update(w.mu, NONIDEAL)[1][:6, :6] for w in wits]
        self.assertGreater(np.abs(cross).max(), 1e-12)
        gap = max(np.abs(Pg[np.ix_(f[:6], f[:6])]-l).max()/np.abs(l).max() for f, l in zip(flat, local))
        self.assertGreater(gap, 0.)
        LOCAL_WITNESSES.append({'test': 'cross_point', 'shared_raw_pixels': len(shared), 'global_P_ij_max_abs': float(np.abs(cross).max()),
                                'within_point_relative_gap_vs_global': float(gap),
                                'qualification': 'local mode reports P_ij = 0 by declared block-diagonal approximation'})


class SimplifiedDifferenceTests(unittest.TestCase):
    def test_retained_reference_correlation_changes_the_answer(self):
        pb = one_point('cpu', NONIDEAL)
        up = pb.update()
        up.solve(10, 1e-12)
        new, s, P, post = up.posterior(pb.refs, [0])
        pb.refs.state[:, 0] = torch.tensor(pb.means[0][6:12])
        mean = torch.tensor(pb.means[0][None, :6])
        prior = torch.tensor(pb.covs[0][None, :6, :6])
        simple = update_points_temporal_simplified(mean, prior, pb.build, pb.f1, 10, 1e-12)
        dx = float((s[0, :3]-simple[0][0, :3]).norm())
        dP = float((P[0]-simple[1][0]).abs().max()/simple[1][0].abs().max())
        self.assertGreater(dP, 1e-3)
        LOCAL_WITNESSES.append({'test': 'local_vs_simplified', 'position_difference_m': dx, 'covariance_relative_difference': dP})
