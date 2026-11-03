"""Shared rigid-part twist (spec 7.5, 7.11, 7.12): exact SE(3) chain rule, prismatic/revolute kinematics, group posterior
vs an independent dense Gaussian oracle, naive-replay falsification, wrong-membership falsification, and group motion
from the actual material-following RGB-D rows on exactly rendered planes."""
import unittest
import numpy as np
import torch
from scipy.linalg import expm
from tests.material_fixtures import Scene, S, FOC, DEVICES, make_filter, fd
from estimator import se3
from estimator.temporal_part_twist import transform, apply, point_jacobian, solve_group, point_covariance, membership_residual_test, material_evaluator

PART_TWIST_WITNESSES = []
DT = 1/59.


def np_apply(xi, dt, X, o):
    """Independent rigid transform: 4x4 scipy matrix exponential of dt * hat(xi), xi = [v, omega], about pivot o."""
    v, w = xi[:3], xi[3:]
    Wx = np.array([[0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0]])
    A = np.zeros((4, 4))
    A[:3, :3], A[:3, 3] = Wx, v
    T = expm(dt*A)
    return (X-o)@T[:3, :3].T+T[:3, 3]+o


def position_evaluator(Y, sigma, X_ref_unused=None):
    """Linear-Gaussian position observations y_i of every member (3 rows padded to 4 with a zero row)."""
    def evaluate(Xm):
        M = len(Xm)
        r = torch.zeros((M, 4), dtype=Xm.dtype)
        r[:, :3] = Xm-Y
        H = torch.zeros((M, 4, 3), dtype=Xm.dtype)
        H[:, :3, :3] = torch.eye(3, dtype=Xm.dtype)
        R = torch.diag_embed(torch.tensor([sigma**2]*3+[1.], dtype=Xm.dtype)).expand(M, 4, 4).clone()
        cost = .5*(r[:, :3]**2).sum(1)/sigma**2
        return {'member': torch.arange(M), 'H': H, 'r': r, 'R': R, 'cost': cost, 'key': torch.arange(M)}
    return evaluate


class PartTwistTests(unittest.TestCase):
    def test_exact_velocity_twist_chain_rule_at_large_rotation(self):
        rng = np.random.default_rng(0)
        X = rng.normal(0, .1, (5, 3))+np.array([.3, -.2, 1.1])
        o = np.array([.25, -.15, 1.05])
        for xi in (np.array([.4, -.2, .3, 12., -8., 15.]), np.array([.9, .1, -.5, 0., 0., 0.]), np.array([0., 0., 0., -20., 4., 9.])):
            Xr = torch.tensor(X)
            to = torch.tensor(o)
            xt = torch.tensor(xi)
            Xt = apply(transform(xt, DT), Xr, to)
            np.testing.assert_allclose(Xt.numpy(), np_apply(xi, DT, X, o), atol=1e-13)
            J = point_jacobian(xt, DT, Xt, to).numpy()
            for i in range(len(X)):
                Jfd = fd(lambda q: np_apply(q, DT, X[i:i+1], o)[0], xi, 1e-5)
                np.testing.assert_allclose(J[i], Jfd, rtol=1e-6, atol=1e-9)
            # an uncentred world-left perturbation would be [I | -[X]_x] dt J_l: wrong whenever o != 0 and omega != 0
            Bu = se3.point_jacobian(Xt)@(DT*se3.left_jacobian(DT*xt))
            if abs(xi[3:]).max() > 0:
                self.assertGreater(float((Bu.numpy()-J).__abs__().max()), 1e-3)
            PART_TWIST_WITNESSES.append({'xi': xi.tolist(), 'rotation_angle_rad': float(np.linalg.norm(xi[3:])*DT),
                                         'jacobian_fd_max_abs': float(max(np.abs(J[i]-fd(lambda q: np_apply(q, DT, X[i:i+1], o)[0], xi, 1e-5)).max() for i in range(len(X))))})

    def test_prismatic_and_revolute_kinematics(self):
        X = torch.tensor(np.random.default_rng(1).normal(0, .1, (6, 3)))
        o = torch.tensor([.1, .2, .3])
        v = torch.tensor([.5, -.2, .1, 0., 0., 0.], dtype=torch.float64)
        Xt = apply(transform(v, DT), X, o)
        np.testing.assert_allclose((Xt-X).numpy(), np.tile(DT*v[:3].numpy(), (6, 1)), atol=1e-15)
        Jt = point_jacobian(v, DT, Xt, o, translation_only=True)
        np.testing.assert_allclose(Jt.numpy(), np.tile(DT*np.eye(3), (6, 1, 1)), atol=0)
        xi = torch.tensor([.2, .1, -.3, 2., -1., 3.], dtype=torch.float64)
        h = 1e-7
        Xh = apply(transform(xi, h), X, o)
        vel = (Xh-X)/h
        want = xi[:3]+torch.cross(xi[3:].expand(6, 3), X-o, dim=1)
        np.testing.assert_allclose(vel.numpy(), want.numpy(), atol=1e-6)

    def test_group_posterior_matches_dense_oracle_and_naive_replay_is_overconfident(self):
        rng = np.random.default_rng(2)
        X = torch.tensor(rng.normal(0, .05, (7, 3))+np.array([0., 0., 1.]))
        o = X.mean(0)
        xi_true = torch.tensor([.3, -.1, .2, 6., -4., 9.], dtype=torch.float64)
        sigma = 1e-4
        Y = apply(transform(xi_true, DT), X, o)+torch.tensor(rng.normal(0, sigma, (7, 3)))
        res = solve_group(position_evaluator(Y, sigma), X, o, DT)
        # independent dense oracle: stack all 21 rows, Gauss-Newton normal equations by QR at the estimate
        xi = res['xi'].numpy()
        Jall = np.concatenate([fd(lambda q, i=i: np_apply(q, DT, X[i:i+1].numpy(), o.numpy())[0], xi, 1e-6) for i in range(7)])/sigma
        Q, Rq = np.linalg.qr(Jall)
        P_or = np.linalg.inv(Rq.T@Rq)
        np.testing.assert_allclose(res['P_xi'].numpy(), P_or, rtol=1e-5, atol=1e-12)
        self.assertLess(float((res['xi']-xi_true).abs().max()), 5*float(res['P_xi'].diagonal().sqrt().max()))
        Pp, J = point_covariance(res['xi'], DT, res['X_t'], o, res['P_xi'])
        Jn = np.stack([fd(lambda q, i=i: np_apply(q, DT, X[i:i+1].numpy(), o.numpy())[0], xi, 1e-6) for i in range(7)])
        full = Jn.reshape(21, 6)@P_or@Jn.reshape(21, 6).T                    # joint posterior of all member positions
        np.testing.assert_allclose(Pp.numpy(), np.stack([full[3*i:3*i+3, 3*i:3*i+3] for i in range(7)]), rtol=1e-4, atol=1e-16)
        cross = (J[0]@res['P_xi']@J[1].T).numpy()
        np.testing.assert_allclose(cross, full[0:3, 3:6], rtol=1e-4, atol=1e-16)
        self.assertGreater(np.abs(cross).max(), 0)
        # naive replay: the group estimate inserted again as an independent observation of each point, on top of its
        # own row -> double counting: variance below the correct marginal
        own = np.eye(3)*sigma**2
        naive = np.linalg.inv(np.linalg.inv(own)+np.linalg.inv(Pp[0].numpy()))
        self.assertLess(np.trace(naive), np.trace(Pp[0].numpy()))
        PART_TWIST_WITNESSES.append({'group_P_vs_QR_rel': float(np.abs(res['P_xi'].numpy()-P_or).max()/np.abs(P_or).max()),
                                     'cross_point_cov_max': float(np.abs(cross).max()), 'naive_trace_over_correct': float(np.trace(naive)/np.trace(Pp[0].numpy()))})

    def test_wrong_membership_drags_static_points_and_is_reported(self):
        rng = np.random.default_rng(3)
        Xm = torch.tensor(rng.normal(0, .05, (8, 3))+np.array([0., 0., 1.]))
        Xs = torch.tensor(rng.normal(0, .05, (4, 3))+np.array([.2, 0., 1.]))
        o = Xm.mean(0)
        xi_true = torch.tensor([.6, 0., 0., 0., 0., 0.], dtype=torch.float64)
        sigma = 1e-4
        Ym = apply(transform(xi_true, DT), Xm, o)
        good = solve_group(position_evaluator(Ym, sigma), Xm, o, DT)
        test_good = membership_residual_test(good['rows'], len(Xm), .01)
        X = torch.cat((Xm, Xs))
        Y = torch.cat((Ym, Xs))                                                # static members observed unmoved
        bad = solve_group(position_evaluator(Y, sigma), X, X.mean(0), DT)
        test_bad = membership_residual_test(bad['rows'], len(X), .01)
        drag = float((bad['X_t'][8:]-Xs).norm(dim=1).max())
        self.assertEqual(test_good['status'], 'GROUP_RESIDUALS_CONSISTENT')
        self.assertEqual(test_bad['status'], 'GROUP_IDENTITY_INVALID')
        self.assertGreater(drag, 1e-3)                                         # the wrong merge visibly moves static points
        PART_TWIST_WITNESSES.append({'wrong_merge_static_drag_m': drag, 'status': test_bad['status'],
                                     'failed_members': test_bad['failed_members'].nonzero().flatten().tolist()})

    def test_group_motion_from_material_rgbd_rows(self):
        PX = 2./FOC
        pixels = [y*S+x for y in range(8, 25, 3) for x in range(8, 25, 3)]
        for device in DEVICES:
            for d, arm in (((3*PX, -1.*PX, .02), True), ((1.3*PX, .7*PX, 0.), False), ((0., 0., 0.), False)):
                with self.subTest(device=device, d=d, translation_only=arm):
                    sc = Scene(d=d)
                    frames = [sc.render(k, device) for k in range(2)]
                    filt, _, base = make_filter(sc, device, pixels)
                    Xr = torch.tensor(base, device=device)
                    members = torch.arange(len(pixels), device=device)
                    ev = material_evaluator(filt.rig, filt.noise, filt.policy, torch.zeros((4, 4), dtype=torch.float64, device=device),
                                            frames[1], {0: frames[0], 1: frames[1]}, filt.refs, Xr, members)
                    res = solve_group(ev, Xr, Xr.mean(0), DT, translation_only=arm)
                    err = float((res['X_t']-(Xr+torch.tensor(d, device=device))).norm(dim=1).max())
                    self.assertLess(err, .02*PX)
                    PART_TWIST_WITNESSES.append({'device': device, 'material_group_d_px': [x/PX for x in d], 'translation_only': arm,
                                                 'max_point_error_px': err/PX, 'rank': res['rank'], 'iterations': len(res['history'])})


if __name__ == '__main__':
    unittest.main()
