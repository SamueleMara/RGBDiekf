"""Material-following local Gaussian vs an independent dense within-point computation (spec section 6.D).

For a traced point the oracle rebuilds the augmented prior over EVERY local variable of the production posterior layout
(current state, source chart or reference state, reference raw errors, current support raw errors), differentiates its
own NumPy material residual (tests/material_fixtures.np_material_f + np_chart) by 5-point finite differences with
respect to all of them, and conditions with formulas independent of the production Joseph update:
null-space projection N (N^T Sigma^-1 N)^-1 N^T for R_warp = 0, information form (Sigma^-1 + H^T R^-1 H)^-1 otherwise.
"""
import unittest
import numpy as np
import torch
from tests.material_fixtures import (Scene, S, FOC, DEVICES, RGB_VAR, DEPTH_VAR, make_filter, run_frames, np_material_f, np_chart, rgbd, fd)
from estimator.temporal_local_covariance import Layout, KEY, STATE, CHART, ForbiddenKeyRecreation, check_no_recreation
from estimator.temporal_reference import ReferenceTable
from estimator.temporal_material_terms import material_record_expiry, material_retention_mask

PX = 2./FOC
MATERIAL_COVARIANCE_WITNESSES = []


def oracle_blocks(step, point):
    """Posterior layout of `point` (production order) and the matching augmented prior mean / covariance."""
    up = step['update']
    tr = step['post']['traces'][point]
    layout = tr['posterior_layout']
    prior_desc = up.prior.layout.describe(point)
    mu_p, Sig_p = (x[0].cpu().numpy() for x in up.prior.gather(torch.tensor([point], device=up.dev)))
    n = layout[-1]['columns'][1]
    mu, Sig = np.zeros(n), np.zeros((n, n))
    key = lambda b: (b['kind'], b['camera'], b['frame'], b['pixel'])
    pidx = {key(b): b['columns'] for b in prior_desc}
    cols_prior, cols_post = [], []
    for b in layout:
        a, e = b['columns']
        if key(b) in pidx:
            pa, pe = pidx[key(b)]
            cols_prior += list(range(pa, pe))
            cols_post += list(range(a, e))
        else:
            if b['kind'] != 'raw_sensor_error':
                raise AssertionError('only fresh raw sensor errors may enter the posterior layout')
            Sig[a:e, a:e] = np.diag([RGB_VAR]*3+[DEPTH_VAR])
    cp, cq = np.array(cols_prior), np.array(cols_post)
    mu[cq] = mu_p[cp]
    Sig[np.ix_(cq, cq)] = Sig_p[np.ix_(cp, cp)]
    return layout, mu, Sig, np.asarray(tr['mean_plus']), np.asarray(tr['Sigma_plus'])


def oracle_residual(step, point, layout, scene, frames, filt):
    """Stacked f(a) for the final rows of `point`, from the oracle only (discrete domain from production rows)."""
    lr = step['lr']
    rows = lr.rows
    ix = torch.where(rows.point == point)[0].tolist()
    find = lambda kind, cam, frame, pix: next(b['columns'] for b in layout if b['kind'] == kind and (cam is None or b['camera'] == cam)
                                              and (frame is None or b['frame'] == frame) and (pix is None or b['pixel'] == pix))
    chart = filt.chart
    def f(a):
        out = []
        xt = a[0:3]
        for r in ix:
            c, rf, pr = int(rows.camera[r]), int(rows.reference_frame[r]), int(rows.reference_pixel[r])
            if rf == chart.anchor_frame:
                ca, _ = find('source_chart', None, None, None)
                sa, _ = find('raw_sensor_error', int(chart.camera[point]), chart.anchor_frame, int(chart.pixel[point]))
                pix0 = int(chart.pixel[point])
                xr = np_chart(a[ca:ca+2], a[sa+3], chart.base[point].cpu().numpy(), np.array([pix0 % S, pix0//S], float),
                              float(chart.depth[point]), scene.t[int(chart.camera[point])])
            else:
                ra, _ = find('state', None, rf, None)
                xr = a[ra:ra+3]
            ka, _ = find('raw_sensor_error', c, rf, pr)
            cell = rows.cell[r].cpu().numpy()
            eta = []
            for dy, dx in ((0, 0), (0, 1), (1, 0), (1, 1)):
                pa, _ = find('raw_sensor_error', c, filt_frame(step), int((cell[1]+dy)*S+cell[0]+dx))
                eta.append(a[pa:pa+4])
            out.append(np_material_f(xt, xr, scene.t[c], pr, rgbd(frames[filt_frame(step)], c), rgbd(frames[rf], c), cell,
                                     eta_cur=np.array(eta), eta_ref=a[ka:ka+4])[0])
        return np.concatenate(out)
    return f


def filt_frame(step):
    return step['update'].k


def null_space_conditioning(Sig, H):
    U, sv, Vt = np.linalg.svd(H)
    N = Vt[np.sum(sv > sv.max()*1e-12):].T
    return N@np.linalg.solve(N.T@np.linalg.solve(Sig, N), N.T)


class MaterialLocalCovarianceTests(unittest.TestCase):
    def run_case(self, device, warp, d):
        sc = Scene(d=d)
        frames = [sc.render(k, device) for k in range(3)]
        filt, _, base = make_filter(sc, device, [16*S+16], warp=warp)
        st = run_frames(filt, frames, 1, traced=[0])[0]
        layout, mu, Sig, a, Splus = oracle_blocks(st, 0)
        f = oracle_residual(st, 0, layout, sc, frames, filt)
        H = fd(f, a, 1e-5)          # 5-point stencil: truncation O(h^4); smaller h is round-off dominated
        if np.abs(warp).max() == 0:
            P_or = null_space_conditioning(Sig, H)
        else:
            R = np.kron(np.eye(H.shape[0]//4), np.asarray(warp))
            P_or = np.linalg.inv(np.linalg.inv(Sig)+H.T@np.linalg.solve(R, H))
        scale = np.sqrt(np.outer(np.diag(Sig), np.diag(Sig)))
        gap = float(np.abs(P_or-Splus).max()/np.abs(Sig).max())
        gap_corr = float(np.abs((P_or-Splus)/scale).max())
        # fixed-prior IEKF stationarity of the published mean at its own linearization
        Rm = np.zeros((H.shape[0],)*2) if np.abs(warp).max() == 0 else np.kron(np.eye(H.shape[0]//4), np.asarray(warp))
        G = Sig@H.T
        a_next = mu+G@np.linalg.solve(H@G+Rm, -f(a)+H@(a-mu))
        step_corr = float(np.abs((a_next-a)/np.sqrt(np.diag(Sig))).max())
        return st, gap, gap_corr, step_corr, H, layout

    def test_posterior_matches_independent_dense_conditioning(self):
        for device in DEVICES:
            for warp, d in ((np.zeros((4, 4)), (1.3*PX, 0., 0.)), (np.diag([1e-5, 1e-5, 1e-5, 1e-6]), (-1.3*PX, .5*PX, .02))):
                with self.subTest(device=device, warp=float(warp.max())):
                    st, gap, gap_corr, step_corr, H, layout = self.run_case(device, warp, d)
                    self.assertLess(gap_corr, 1e-6)
                    self.assertLess(step_corr, 1e-3)
                    kinds = [b['kind'] for b in layout]
                    self.assertEqual(kinds.count('raw_sensor_error'), 1+1+2*4)   # cam-0 reference = shared source key, cam-1 reference key, 4+4 current
                    MATERIAL_COVARIANCE_WITNESSES.append({'device': device, 'warp_max': float(warp.max()), 'posterior_gap_correlation_units': gap_corr,
                                                          'posterior_gap_relative': gap, 'iekf_fixed_point_step_prior_sigmas': step_corr,
                                                          'local_dimension': len(H[0]), 'rows': H.shape[0]//4})

    def test_source_key_shared_and_reference_key_persists(self):
        """Camera-0 reference at frame 0 IS the source pixel: its raw error is the chart's source error (reused, not a
        second copy). After frame 1 exactly one raw key per committed camera is retained -- the new reference pixel --
        correlated with the state, and frame 2 reuses it."""
        sc = Scene(d=(1.3*PX, 0., 0.))
        frames = [sc.render(k, 'cpu') for k in range(3)]
        filt, _, base = make_filter(sc, 'cpu', [16*S+16])
        st1 = run_frames(filt, frames, 1)[0]
        self.assertEqual(st1['lr'].reused_source_keys, 1)
        L = filt.bank.layout
        keys = torch.where((L.point == 0) & (L.kind == KEY))[0]
        for j in keys.tolist():
            c = int(L.camera[j])
            self.assertEqual(int(L.frame[j]), 1)
            self.assertEqual(int(L.pixel[j]), int(filt.refs.pixel[c, 0]))
        self.assertEqual(len(keys), int((filt.refs.frame[:, 0] == 1).sum()))
        mean, cov = filt.bank.gather(torch.tensor([0]))
        start = int(L.start[keys[0]])
        self.assertGreater(float(cov[0, :6, start:start+4].abs().max()), 0.)
        st2 = filt.step({j: frames[j] for j in range(3)}, 2, 2, .04, 10, 1e-10)
        self.assertEqual(st2['lr'].reused_history_keys, len(keys))
        MATERIAL_COVARIANCE_WITNESSES.append({'reused_source_keys_frame1': st1['lr'].reused_source_keys, 'retained_reference_keys': len(keys),
                                              'reused_history_keys_frame2': st2['lr'].reused_history_keys})

    def test_expiry_ledger_blocks_only_needed_keys(self):
        """Expiring a support key of the reference frame at another pixel does not block a never-used raw pixel; expiring
        the reference pixel key (or an older frame) blocks its recreation."""
        refs = ReferenceTable(1, 1, 'cpu', torch.float64)
        refs.frame[0, 0], refs.ordinal[0, 0], refs.pixel[0, 0] = 3, 3, 40
        fo = torch.full((8,), -1, dtype=torch.long)
        fo[2], fo[3] = 2, 3
        t = lambda *v: torch.tensor(v)
        layout = Layout(1, t(0, 0, 0), t(KEY, KEY, KEY), t(0, 0, 0), t(3, 3, 2), t(41, 40, 40))
        keep = torch.tensor([False, True, False])
        exp = torch.full((1, 1), -1, dtype=torch.long)
        material_record_expiry(layout, keep, refs, exp, fo)
        self.assertEqual(int(exp[0, 0]), 2)                   # frame-2 key recorded; frame-3 support pixel 41 not recorded
        check_no_recreation(t(0), t(0), t(3), exp, fo)        # frame 3 fresh pixel allowed
        with self.assertRaises(ForbiddenKeyRecreation):
            check_no_recreation(t(0), t(0), t(2), exp, fo)
        keep = torch.tensor([False, False, False])
        material_record_expiry(layout, keep, refs, exp, fo)   # the reference pixel key itself expired
        self.assertEqual(int(exp[0, 0]), 3)

    def test_backward_processing(self):
        """Catalogue at the last frame, processing 2 -> 1 -> 0 with negative dt (B1 direction)."""
        d = (1.3*PX, 0., 0.)
        sc = Scene(d=d)
        frames = [sc.render(k, 'cpu') for k in range(3)]
        from estimator.temporal_material_run import MaterialFilter
        from tests.material_fixtures import rig, noise, POLICY
        pix = [16*S+16, 14*S+18]
        src = np.array([[2, 0, p] for p in pix])
        pts2 = []
        for p in pix:
            lam = float(frames[2].depth[0, p//S, p % S])
            pts2.append(np.array([lam*(p % S-15.5)/FOC, lam*(p//S-15.5)/FOC, lam]))
        filt = MaterialFilter(rig('cpu'), noise('cpu'), POLICY, np.zeros((4, 4)), np.eye(3)*1e-4, 3, 10**8, 8, frames[2], src, np.array(pts2))
        filt.initialize(frames[2], np.eye(2)/12, np.eye(3))
        st1 = filt.step({2: frames[2], 1: frames[1]}, 1, 1, -.04)
        st0 = filt.step({2: frames[2], 1: frames[1], 0: frames[0]}, 0, 2, -.04)
        for k, st in ((1, st1), (0, st0)):
            err = np.linalg.norm(st['s'][:, :3].numpy()-(np.array(pts2)-(2-k)*np.asarray(d)), axis=1)
            self.assertLess(float(err.max()), .05*PX)


if __name__ == '__main__':
    unittest.main()
