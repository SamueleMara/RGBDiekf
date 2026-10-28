"""Gate G (decision logic): fixed-axis reduction and the static / prismatic / revolute decision on analytic pose
sequences with CALIBRATED pose noise (drawn from the covariance handed to the decision), variable joint speed and
both processing directions; constant axis over time; the forward/backward comparison never fuses."""
import unittest
import numpy as np
import torch
from estimator import se3
from joint_reduction.joint_projective import reduce_and_decide, axis_constancy, joint_transform, compare_directions

ALPHA = .01


def sequence(kind, K=30, anchor=0, sigma=(2e-4, 1e-3), seed=0):
    g = np.random.default_rng(seed)
    t = np.arange(K)/30.
    q = .25*(1-np.cos(np.pi*np.arange(K)/(K-1)))               # variable speed, zero at both ends
    q = q-q[anchor]
    axis, point = np.array([.3, -.8, .5])/np.linalg.norm([.3, -.8, .5]), np.array([.1, .2, -.05])
    cov = np.diag([sigma[0]**2]*3+[sigma[1]**2]*3)
    poses, covs = np.zeros((K, 4, 4)), np.zeros((K, 12, 12))
    for k in range(K):
        T = np.eye(4) if kind == 'STATIC' else joint_transform(kind, axis, point, q[k] if kind == 'PRISMATIC' else 2*q[k])
        if k != anchor:
            T = se3.exp(torch.as_tensor(g.multivariate_normal(np.zeros(6), cov))).numpy()@T
            covs[k, :6, :6] = cov
        covs[k, 6:, 6:] = np.eye(6)
        poses[k] = T
    core = g.normal(0, .1, (500, 3))
    return poses, covs, t, core, axis, point


class JointDecisionTest(unittest.TestCase):
    def test_types_both_directions(self):
        for kind in ('STATIC', 'PRISMATIC', 'REVOLUTE'):
            for anchor in (0, 29):
                poses, covs, t, core, axis, point = sequence(kind, anchor=anchor, seed=hash((kind, anchor)) % 1000)
                red, tests, status, reason, cand = reduce_and_decide(poses, covs, t, core, anchor, ALPHA)
                self.assertEqual(status, kind, f'{kind} anchor {anchor}: {reason} {tests}')
                if kind != 'STATIC':
                    e = np.asarray(cand[kind]['axis'])
                    self.assertLess(np.degrees(np.arccos(min(1, abs(e@axis)))), 1.)
                    self.assertAlmostEqual(cand[kind]['q'][anchor], 0., places=9)       # gauge q(anchor) = 0
                    if kind == 'REVOLUTE':
                        a = np.asarray(cand[kind]['axis_point']); d = a-point
                        self.assertLess(np.linalg.norm(d-(d@axis)*axis), .01)
                        incr, summ = axis_constancy(poses, t, cand, anchor)
                        self.assertLess(summ['REVOLUTE']['median_deg'], 5.)

    def test_weak_excitation_is_not_called_revolute(self):
        poses, covs, t, core, axis, point = sequence('REVOLUTE', sigma=(2e-4, 1e-3))
        small = poses.copy()
        for k in range(len(small)):     # shrink the motion to the noise level
            xi = se3.log(torch.as_tensor(poses[k])).numpy()*1e-3
            small[k] = se3.exp(torch.as_tensor(xi)).numpy()
        red, tests, status, reason, cand = reduce_and_decide(small, covs, t, core, 0, ALPHA)
        self.assertIn(status, ('STATIC', 'UNIDENTIFIABLE'))

    def test_bidirectional_comparison_does_not_fuse(self):
        pf, cf, t, core, axis, point = sequence('PRISMATIC', anchor=0, seed=3)
        pb, cb, *_ = sequence('PRISMATIC', anchor=29, seed=4)
        jf = dict(zip(('red', 'tests', 'status', 'reason', 'candidates'), reduce_and_decide(pf, cf, t, core, 0, ALPHA)))
        jb = dict(zip(('red', 'tests', 'status', 'reason', 'candidates'), reduce_and_decide(pb, cb, t, core, 29, ALPHA)))
        out = compare_directions(jf, jb)
        self.assertLess(out['per_type']['PRISMATIC']['axis_angle_deg_up_to_sign'], 1.)
        self.assertIn('never fused', out['note'])


if __name__ == '__main__':
    unittest.main()
