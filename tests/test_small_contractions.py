"""STEP 2B-B1 parity: the opt-in broadcast contractions equal the original batched products (OFF bitwise, ON to round-off)."""
import unittest
import torch
from tests.test_image_iekf import small_problem, observation
from pointwise_rgbd import small_contractions as sc
from pointwise_rgbd.config import load_config
from pointwise_rgbd.rgbd_point_renderer import RenderPolicy, render
from pointwise_rgbd.image_iekf import IEKFConfig, frame_step
from pointwise_rgbd.supernodes import GroupBasis, build_supernodes
from pointwise_rgbd.process6 import predict

DEVICES = ['cpu']+(['cuda'] if torch.cuda.is_available() else [])
rel = lambda a, b: float((a-b).norm()/b.norm().clamp_min(1e-300))


class HelperTest(unittest.TestCase):
    def test_off_is_original_expression_on_is_roundoff_equal(self):
        g = torch.Generator().manual_seed(0)
        for dev in DEVICES:
            r = lambda *s: torch.randn(*s, generator=g, dtype=torch.float64).to(dev)
            A, x, y, P, Pk, B = r(500, 5, 3), r(500, 3), r(500, 5), r(5, 5), r(500, 5, 5), r(500, 5, 4)
            original = {'matvec': (A@x[..., None])[..., 0], 'rmatvec': (A.transpose(1, 2)@y[..., None])[..., 0],
                        'congruence': A.transpose(1, 2)@P@A, 'congruence_batched': A.transpose(1, 2)@Pk@A, 'gram': A.transpose(1, 2)@B}
            for flag in (False, True):
                with sc.enabled(flag):
                    got = {'matvec': sc.matvec(A, x), 'rmatvec': sc.rmatvec(A, y), 'congruence': sc.congruence(A, P),
                           'congruence_batched': sc.congruence(A, Pk), 'gram': sc.gram(A, B)}
                for k in original:
                    if flag: self.assertLess(rel(got[k], original[k]), 1e-14, (dev, k))
                    else: self.assertTrue(torch.equal(got[k], original[k]), (dev, k))
        self.assertFalse(sc.is_on(), 'the flag must be restored after the context')


class OperatorParityTest(unittest.TestCase):
    def test_group_basis_and_linearization_on_equals_off(self):
        config = load_config()
        ds, xyz, rgb, rho, G, C, s0, P0 = small_problem(n_max=200, groups=8)
        g = torch.Generator().manual_seed(1)
        z = torch.randn(G.count, 12, generator=g, dtype=torch.float64); f = torch.randn(len(xyz), 6, generator=g, dtype=torch.float64)
        M = torch.randn(len(xyz), 6, 6, generator=g, dtype=torch.float64)
        lin = render(s0[:, :3], rgb, rho, C, RenderPolicy()); terms = observation(ds, 1, config).terms(lin)
        v = torch.randn(len(xyz), 3, generator=g, dtype=torch.float64)
        a = [torch.randn(len(r_), 5, generator=g, dtype=torch.float64) for r_ in terms['residual_adjoint']]
        out = {}
        for flag in (False, True):
            with sc.enabled(flag):
                out[flag] = {'expand': G.expand(z), 'reduce': G.reduce(f), 'blocks': G.blocks(M), 'H': torch.cat(lin.apply_H(v)),
                             'HT': lin.apply_HT(a), 'fisher': lin.block_fisher(terms['weights'])}
                ze = G.expand(z); self.assertLess(abs(float((ze*f).sum()-(z*G.reduce(f)).sum()))/float(ze.norm()*f.norm()), 1e-14)
        for k in out[False]:
            self.assertLess(rel(out[True][k], out[False][k]), 1e-13, k)
        self.assertLess(rel(out[True]['blocks'], GroupBasis.blocks(G, M)), 1e-13)   # outside the context: the original path

    def test_frame_update_on_equals_off(self):
        config = load_config()
        ds, xyz, rgb, rho, G, C, s0, P0 = small_problem()
        sm, Pm = predict(s0, P0, 1/30, config)
        res = {}
        for flag in (False, True):
            res[flag] = frame_step(sm, Pm.clone(), [(observation(ds, 1, config), C, 1.)], rgb, rho, RenderPolicy(), G, IEKFConfig(small_contractions=flag))
        (s0_, P0_, *_, r0), (s1_, P1_, *_, r1) = res[False], res[True]
        self.assertEqual([h['accepted'] for h in r0['history']], [h['accepted'] for h in r1['history']])
        self.assertLess(rel(s1_-sm, s0_-sm), 1e-9)
        self.assertLess(rel(P1_, P0_), 1e-9)
        self.assertFalse(sc.is_on())


if __name__ == '__main__':
    unittest.main()
