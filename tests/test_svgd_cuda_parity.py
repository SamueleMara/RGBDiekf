"""TF.6 item 8 / Gate 8: CPU float64 vs tiled CUDA float64 for every operative operator, per-operator max / p95 discrepancy.
Tolerance basis: both sides are float64 with different reduction orders and tile shapes; the relative error is bounded by
~(number of summed terms) x eps x condition, here <= 1e-10 (measured values are recorded)."""
import unittest
import numpy as np
import torch
from estimator.svgd_target_score import target_score
from estimator.svgd_particle_operator import stein_field
from estimator.svgd_feature_measurement import RFFFeatures
from estimator.svgd_temporal import TemporalPrior
from estimator.svgd_rigidity import knn_graph, edge_residuals, edge_covariance_structured, rigid_edge_loglik
from tests.svgd_fixtures import record

CUDA = torch.cuda.is_available()
TOL = 1e-10


def cmp(name, eq, a, b):
    a, b = a.detach().cpu().double(), b.detach().cpu().double(); scale = float(a.abs().max())+1e-300
    d = (a-b).abs()/scale
    out = {'max_rel': float(d.max()), 'p95_rel': float(torch.quantile(d.reshape(-1)[:2**24], .95))}
    record(eq, test=f'cuda_parity_{name}', device=torch.cuda.get_device_name(0), tolerance=TOL, **out)
    return out


@unittest.skipUnless(CUDA, 'CUDA unavailable')
class CudaParityTests(unittest.TestCase):
    def setUp(self):
        torch.backends.cuda.matmul.allow_tf32 = False
        g = np.random.default_rng(0); self.N, self.M = 2000, 3000
        self.x = torch.tensor(g.normal(0, .02, (self.N, 3))); self.c = torch.tensor(g.uniform(0, 1, (self.N, 3)))
        self.y = torch.tensor(g.normal(0, .02, (self.M, 3))); self.d = torch.tensor(g.uniform(0, 1, (self.M, 3)))
        A = g.normal(size=(self.M, 3, 3)); self.LX = torch.tensor(np.einsum('nij,nkj->nik', A, A)*1e-6+np.eye(3)*1e-6); self.LC = torch.eye(3, dtype=torch.float64)*.01

    def test_target_score(self):
        cpu = target_score(self.x, self.c, self.y, self.d, self.LX, self.LC, workspace_bytes=32*2**20)
        gpu = target_score(*(t.cuda() for t in (self.x, self.c, self.y, self.d, self.LX, self.LC)), workspace_bytes=8*2**20)
        for nm, a, b in (('log_density', cpu.log_density, gpu.log_density), ('grad_xyz', cpu.grad_xyz, gpu.grad_xyz)):
            self.assertLess(cmp(nm, 'E03', a, b)['max_rel'], TOL)

    def test_stein_field(self):
        g = target_score(self.x, self.c, self.y, self.d, self.LX, self.LC).grad_xyz; H = torch.eye(3, dtype=torch.float64)*1e-4
        cpu = stein_field(self.x, self.c, g, H, self.LC, workspace_bytes=32*2**20); gpu = stein_field(self.x.cuda(), self.c.cuda(), g.cuda(), H, self.LC, workspace_bytes=8*2**20)
        for nm in ('attraction', 'repulsion', 'total', 'kernel_mass'): self.assertLess(cmp(nm, 'E07', getattr(cpu, nm), getattr(gpu, nm))['max_rel'], TOL)

    def test_feature_operator(self):
        u = torch.randn(self.N, 3, dtype=torch.float64, generator=torch.Generator().manual_seed(1)); w = torch.randn(512, dtype=torch.float64, generator=torch.Generator().manual_seed(2))
        fc, fg = RFFFeatures(torch.eye(3)*1e-4, torch.eye(3)*.01, 512, 3), RFFFeatures(torch.eye(3)*1e-4, torch.eye(3)*.01, 512, 3, device='cuda')
        self.assertEqual(fc.digest(), fg.digest())
        for nm, a, b in (('mean', fc.mean(self.x, self.c), fg.mean(self.x.cuda(), self.c.cuda())), ('jvp', fc.jvp(self.x, self.c, u), fg.jvp(self.x.cuda(), self.c.cuda(), u.cuda())),
                         ('vjp', fc.vjp(self.x, self.c, w), fg.vjp(self.x.cuda(), self.c.cuda(), w.cuda())), ('covariance', fc.covariance(self.x, self.c), fg.covariance(self.x.cuda(), self.c.cuda()))):
            self.assertLess(cmp(nm, 'E13', a, b)['max_rel'], TOL)

    def test_temporal_and_rigidity(self):
        times = [0., .03, .07, .1]; P0 = torch.eye(3, dtype=torch.float64).expand(self.N, 3, 3)*1e-6
        tc, tg = TemporalPrior(times, np.eye(3)*.2, self.x, P0, np.eye(3)*.01), TemporalPrior(times, np.eye(3)*.2, self.x.cuda(), P0.cuda(), np.eye(3)*.01)
        R = torch.randn(self.N, 4, 6, dtype=torch.float64, generator=torch.Generator().manual_seed(3))
        self.assertLess(cmp('temporal_solve', 'E18', tc.solve(R), tg.solve(R.cuda()))['max_rel'], TOL)
        self.assertLess(cmp('temporal_hess', 'E18', tc.hess_apply(R), tg.hess_apply(R.cuda()))['max_rel'], TOL)
        X = self.x[None]+torch.randn(4, self.N, 3, dtype=torch.float64, generator=torch.Generator().manual_seed(4))*1e-3
        ec, _ = knn_graph(self.x, 8); eg, _ = knn_graph(self.x.cuda(), 8)
        self.assertTrue(torch.equal(ec, eg.cpu()))
        P = torch.eye(3, dtype=torch.float64).expand(4, self.N, 3, 3)*1e-7
        rc = rigid_edge_loglik(edge_residuals(X, ec)[0], *edge_covariance_structured(X, ec, P))[0]
        rg = rigid_edge_loglik(edge_residuals(X.cuda(), eg)[0], *edge_covariance_structured(X.cuda(), eg, P.cuda()))[0]
        self.assertLess(cmp('rigid_loglik', 'E24', rc, rg)['max_rel'], TOL)


if __name__ == '__main__':
    unittest.main()
