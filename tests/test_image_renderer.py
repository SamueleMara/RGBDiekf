"""Gate A: colored point-cloud RGB-D rasterizer and its exact H / H^T."""
import unittest
import numpy as np
import torch
from tests.image_fixtures import two_plate_scene, catalog
from pointwise_rgbd.rgbd_point_renderer import RenderPolicy, CameraTensors, render, reference_render

DEVICES = ['cpu']+(['cuda'] if torch.cuda.is_available() else [])


def setup(device, motion='prismatic', offset=.06, cameras=3):
    ds = two_plate_scene(motion, frames=3, cameras=cameras, width=64, height=48, offset=offset)
    xyz, rgb, rho, _ = catalog(ds, 0)
    t = lambda a: torch.as_tensor(a, dtype=torch.float64, device=device)
    cams = [CameraTensors(c, torch.device(device)) for c in ds.cameras]
    # Exact ray-cast depth: declare 1 mm depth noise; the 6 cm plate offset exceeds noise + footprint slope.
    return ds, t(xyz), t(rgb), t(rho), cams, RenderPolicy(depth_variance=1e-6)


class RendererTest(unittest.TestCase):
    def test_vectorized_render_matches_loop_oracle(self):
        for device in DEVICES:
            ds, x, c, r, cams, policy = setup(device)
            lin = render(x, c, r, cams, policy)
            for cam, view in zip(ds.cameras, lin.views):
                rgb, depth, cov, coverage = reference_render(x.cpu(), c.cpu(), r.cpu(), cam, policy)
                self.assertTrue(np.array_equal(cov, view.covered.cpu().numpy()), device)
                self.assertLess(np.abs(rgb-view.rgb.cpu().numpy()).max(), 1e-12)
                self.assertLess(np.abs(depth-view.depth.cpu().numpy()).max(), 1e-12)
                self.assertLess(np.abs(coverage-view.coverage.cpu().numpy()).max(), 1e-12)
                self.assertGreater(cov.sum(), 100)

    def test_cpu_cuda_agree(self):
        if len(DEVICES) < 2: self.skipTest('CUDA unavailable')
        a = setup('cpu'); b = setup('cuda')
        la, lb = render(*a[1:4], a[4], a[5]), render(*b[1:4], b[4], b[5])
        dx = torch.randn(len(a[1]), 3, dtype=torch.float64, generator=torch.Generator().manual_seed(1))
        for va, vb, ha, hb in zip(la.views, lb.views, la.apply_H(dx), lb.apply_H(dx.cuda())):
            self.assertTrue(torch.equal(va.covered, vb.covered.cpu()))
            self.assertLess((va.rgb-vb.rgb.cpu()).abs().max().item(), 1e-12)
            self.assertLess((ha-hb.cpu()).abs().max().item(), 1e-9)

    def test_jvp_matches_finite_difference_of_rerender(self):
        for device in DEVICES:
            ds, x, c, r, cams, policy = setup(device)
            lin = render(x, c, r, cams, policy)
            g = torch.Generator().manual_seed(2)
            dx = torch.randn(len(x), 3, dtype=torch.float64, generator=g).to(device)*1e-3
            eps = 1e-5
            plus, minus = render(x+eps*dx, c, r, cams, policy), render(x-eps*dx, c, r, cams, policy)
            for v, vp, vm, h in zip(lin.views, plus.views, minus.views, lin.apply_H(dx)):
                same = v.covered & vp.covered & vm.covered
                fd = torch.cat(((vp.rgb-vm.rgb)/(2*eps), ((vp.depth-vm.depth)/(2*eps))[:, None],
                                ((vp.coverage-vm.coverage)/(2*eps))[:, None]), 1)[same]
                err = (fd-h[same]).abs().max()/fd.abs().max()
                self.assertLess(err.item(), 1e-4, device)

    def test_adjoint_identity(self):
        for device in DEVICES:
            ds, x, c, r, cams, policy = setup(device)
            lin = render(x, c, r, cams, policy)
            g = torch.Generator().manual_seed(3)
            v = torch.randn(len(x), 3, dtype=torch.float64, generator=g).to(device)
            a = [torch.randn(cm.width*cm.height, 5, dtype=torch.float64, generator=g).to(device) for cm in cams]
            lhs = sum((hv*ai).sum() for hv, ai in zip(lin.apply_H(v), a))
            rhs = (v*lin.apply_HT(a)).sum()
            self.assertLess(abs(lhs-rhs).item()/abs(lhs).item(), 1e-12)

    def test_metric_depth_units_against_raycast(self):
        ds, x, c, r, cams, policy = setup('cpu')
        lin = render(x, c, r, cams, policy)
        for ci, v in enumerate(lin.views):
            f = ds.observation(0, ci)
            use = v.covered.numpy() & f.valid.ravel()
            err = np.abs(v.depth.numpy()[use]-f.depth.ravel()[use])
            # Metric agreement within the model's own resolution (footprint radius);
            # a unit error (mm vs m) would be off by orders of magnitude.
            self.assertLess(np.median(err), np.median(r.numpy()))
            self.assertLess(np.median(err), 1e-2)

    def test_rgb_sensitivity_for_in_plane_textured_motion(self):
        ds, x, c, r, cams, policy = setup('cpu')
        lin = render(x, c, r, cams, policy)
        dx = torch.zeros_like(x); dx[:, 0] = 1.   # slide every sample along the textured plane
        h = lin.apply_H(dx)
        rgb_change = sum(int((hv[v.covered, :3].abs().sum(1) > 1e-3).sum()) for hv, v in zip(h, lin.views))
        self.assertGreater(rgb_change, 200)

    def test_fisher_blocks_equal_dense_diagonal(self):
        ds, x, c, r, cams, policy = setup('cpu')
        x, c, r = x[:300], c[:300], r[:300]
        lin = render(x, c, r, cams, policy)
        weights = [(torch.ones(cm.width*cm.height, dtype=torch.float64), torch.eye(3, dtype=torch.float64),
                    torch.ones(cm.width*cm.height, dtype=torch.float64), torch.ones(cm.width*cm.height, dtype=torch.float64)) for cm in cams]
        F = lin.block_fisher(weights)
        for i in range(0, 300, 37):
            cols = []
            for k in range(3):
                e = torch.zeros_like(x); e[i, k] = 1
                cols.append(torch.cat([h.reshape(-1) for h in lin.apply_H(e)]))
            J = torch.stack(cols, 1)
            self.assertLess((J.T@J-F[i]).abs().max().item(), 1e-9)


class DeterminismTest(unittest.TestCase):
    def test_repeated_render_and_operators_are_bitwise_identical(self):
        for device in DEVICES:
            ds, x, c, r, cams, policy = setup(device, offset=.03)
            a, b = render(x, c, r, cams, policy), render(x, c, r, cams, policy)
            v = torch.randn(len(x), 3, dtype=torch.float64, generator=torch.Generator().manual_seed(7)).to(device)
            for va, vb, ha, hb in zip(a.views, b.views, a.apply_H(v), b.apply_H(v)):
                self.assertTrue(torch.equal(va.rgb, vb.rgb) and torch.equal(va.depth, vb.depth) and torch.equal(va.coverage, vb.coverage))
                self.assertTrue(torch.equal(ha, hb))
            z = [torch.ones(cm.width*cm.height, 5, dtype=torch.float64, device=device) for cm in cams]
            self.assertTrue(torch.equal(a.apply_HT(z), b.apply_HT(z)))


class VisibilityTransitionTest(unittest.TestCase):
    """Contract section 3: ordering boundaries, overlap, footprint edges; all cameras at once."""
    def test_render_is_continuous_along_ordering_band_and_edge_transitions(self):
        policy = RenderPolicy(depth_variance=1e-6)
        ds, x, c, r, cams, _ = setup('cpu', offset=.03)
        # Drive the moving plate THROUGH the base along the camera-facing normal and across its edge: every sample
        # crosses the band, changes depth order with the base, and footprints leave/enter pixels.
        moving = x[:, 2] > .015
        path = torch.zeros_like(x); path[moving] = torch.tensor([.03, .01, -.05], dtype=torch.float64)
        def frame(t):
            lin = render(x+t*path, c, r, cams[:1], policy); v = lin.views[0]
            return torch.cat([v.rgb, v.depth[:, None], v.coverage[:, None]], 1)
        ts = torch.linspace(0, 1, 401, dtype=torch.float64); jumps = []
        prev = frame(ts[0])
        for t in ts[1:]:
            cur = frame(t); jumps.append(float((cur-prev).abs().max())); prev = cur
        fine = max(float((frame(t+1e-7)-frame(t)).abs().max()) for t in ts[::20])
        self.assertLess(fine, 1e-3, 'a 1e-7 path step must not jump (hard cuts gave O(0.1-1) jumps)')
        self.assertGreater(max(jumps), 1e-3)                    # the path does change the image

    def test_jvp_with_tapered_overlap_matches_finite_difference(self):
        for device in DEVICES:
            ds, x, c, r, cams, policy = setup(device, offset=.03)
            lin = render(x, c, r, cams, policy)
            tapered = 0
            for v in lin.views:
                tapered += int((v.weight < (v.weight.new_ones(1))).sum())
            g = torch.Generator().manual_seed(5)
            dx = torch.randn(len(x), 3, dtype=torch.float64, generator=g).to(device)*1e-3
            eps = 1e-6
            plus, minus = render(x+eps*dx, c, r, cams, policy), render(x-eps*dx, c, r, cams, policy)
            for v, vp, vm, h in zip(lin.views, plus.views, minus.views, lin.apply_H(dx)):
                fd = torch.cat(((vp.rgb-vm.rgb)/(2*eps), ((vp.depth-vm.depth)/(2*eps))[:, None], ((vp.coverage-vm.coverage)/(2*eps))[:, None]), 1)
                use = v.total > 0
                err = (fd-h)[use].abs().max()/fd[use].abs().max()
                self.assertLess(err.item(), 1e-4, device)

    def test_front_depth_derivative_reaches_occluder(self):
        """Moving only the occluding plate along depth changes the occluded plate's visibility (through d v / d zeta_front)."""
        policy = RenderPolicy(depth_variance=1e-6)
        ds, x, c, r, cams, _ = setup('cpu', offset=.03)
        lin = render(x, c, r, cams, policy)
        front = x[:, 2] > .015
        dx = torch.zeros_like(x); dx[front, 2] = 1.
        h = lin.apply_H(dx)
        eps = 1e-7                                     # central-difference truncation error ~ eps^2 (checked: 1e-5 -> 1e-7 -> 1e-9)
        p, m = render(x+eps*dx, c, r, cams, policy), render(x-eps*dx, c, r, cams, policy)
        for v, vp, vm, hv in zip(lin.views, p.views, m.views, h):
            fd = (vp.coverage-vm.coverage)/(2*eps)
            self.assertLess(float((fd-hv[:, 4]).abs().max()), 1e-5*max(1., float(fd.abs().max())))

    def test_all_eight_cameras_adjoint(self):
        ds, x, c, r, cams, policy = setup('cpu', cameras=8)
        lin = render(x, c, r, cams, policy)
        self.assertEqual(len(lin.views), 8)
        g = torch.Generator().manual_seed(9)
        v = torch.randn(len(x), 3, dtype=torch.float64, generator=g)
        a = [torch.randn(cm.width*cm.height, 5, dtype=torch.float64, generator=g) for cm in cams]
        lhs = sum((hv*ai).sum() for hv, ai in zip(lin.apply_H(v), a)); rhs = (v*lin.apply_HT(a)).sum()
        self.assertLess(abs(lhs-rhs).item()/abs(lhs).item(), 1e-12)
        # H^T sums every camera: equals the sum of single-camera adjoints
        single = sum(render(x, c, r, [cm], policy).apply_HT([ai]) for cm, ai in zip(cams, a))
        self.assertLess(float((single-lin.apply_HT(a)).abs().max()), 1e-10)


class PyramidTest(unittest.TestCase):
    def test_coarse_level_aligns_with_block_averaged_observation(self):
        from pointwise_rgbd.image_observation import downsample
        ds, x, c, r, cams, policy = setup('cpu')
        for ci, cam in enumerate(ds.cameras):
            f = downsample(ds.observation(0, ci), 2)
            self.assertEqual((f.camera.width, f.camera.height), (cam.width//2, cam.height//2))
            v = render(x, c, 2*r, [CameraTensors(f.camera, torch.device('cpu'))], policy).views[0]
            use = v.covered.numpy() & f.valid.ravel()
            err = np.abs(v.depth.numpy()[use]-f.depth.ravel()[use])
            self.assertGreater(use.sum(), 50)
            self.assertLess(np.median(err), np.median(2*r.numpy()))
            # A half-pixel principal-point error would shift the silhouette: covered vs domain overlap stays high.
            self.assertGreater(use.sum()/f.valid.sum(), .9)


if __name__ == '__main__':
    unittest.main()
