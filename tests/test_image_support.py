"""Contract section 11 gates: global-label support, exact fast path, exact enumeration; bidirectional diagnostics."""
import itertools, tempfile, unittest
import numpy as np
import torch
from tests.image_fixtures import two_plate_scene, catalog, source_parts
from scene.config import load_config
from estimator.point_renderer import CameraTensors, RenderPolicy, render
from estimator.observation_terms import FrameObservation, ObservationPolicy
from joint_reduction.support import delta_flips, total_energy, block_icm, scene_points
from joint_reduction.bidirectional import compare
from estimator.joint_iekf import joint_transform

DEVICES = ['cpu']+(['cuda'] if torch.cuda.is_available() else [])
MODEL = {'type': 'PRISMATIC', 'axis': (np.array([1., .3, 0.])/np.linalg.norm([1., .3, 0.])).tolist()}


def setup(device, frames=4, cameras=2, width=56, height=42):
    ds = two_plate_scene('prismatic', frames=frames, cameras=cameras, width=width, height=height)
    xyz, rgb, rho, src = catalog(ds, 0)
    t = lambda a: torch.as_tensor(np.asarray(a), dtype=torch.float64, device=device)
    cams = [CameraTensors(cm, torch.device(device)) for cm in ds.cameras]
    q = np.linalg.norm([1., .3, 0.])*.02*np.arange(frames)
    return ds, t(xyz), t(rgb), t(rho), source_parts(ds, src), cams, q


class SupportFastPathTest(unittest.TestCase):
    def test_flip_differences_equal_complete_rerender_after_label_changes(self):
        config = load_config(); policy = RenderPolicy(depth_variance=1e-6)
        for device in DEVICES:
            ds, p, c, r, parts, cams, q = setup(device)
            g = np.random.default_rng(0)
            moving = torch.as_tensor(parts == 1, device=device)
            moving[torch.nonzero(moving).flatten()[::3]] = False                       # some movers currently static
            moving[torch.as_tensor(g.choice(np.flatnonzero(parts == 0), 40, replace=False), device=device)] = True   # wrong labels
            for k in (1, 3):
                J = joint_transform(MODEL, float(q[k])).to(device)
                base, alt = scene_points(p, moving, J)
                active = torch.zeros(len(p), dtype=torch.bool, device=device)
                picks = torch.as_tensor(np.r_[g.choice(np.flatnonzero(parts == 1), 8), g.choice(np.flatnonzero(parts == 0), 8)], device=device)
                active[picks] = True
                for ci, cam in enumerate(cams):
                    obs = FrameObservation([ds.observation(k, ci)], config, ObservationPolicy(), device)
                    d, n = delta_flips(cam, 0, obs, base, alt, active, c, r, policy)
                    e0 = obs.terms(render(base, c, r, [cam], policy))['energy']
                    gone = torch.full_like(base, float('nan'))
                    dr, _ = delta_flips(cam, 0, obs, base, gone, active, c, r, policy)
                    for i in picks.tolist():
                        x = base.clone(); x[i] = alt[i]
                        e1 = obs.terms(render(x, c, r, [cam], policy))['energy']
                        self.assertAlmostEqual(d[i].item(), e1-e0, delta=1e-8*max(1., abs(e0)), msg=f'{device} frame {k} cam {ci} point {i}')
                        x = base.clone(); x[i] = float('nan')                                   # absent
                        e2 = obs.terms(render(x, c, r, [cam], policy))['energy']
                        self.assertAlmostEqual(dr[i].item(), e2-e0, delta=1e-8*max(1., abs(e0)), msg=f'removal {device} frame {k} cam {ci} point {i}')


class ExactEnumerationTest(unittest.TestCase):
    """The ICM MAP vs the exact MAP over all 2^n labellings of n coupled candidates (complete-scene energies)."""
    def test_icm_reaches_exact_map_on_small_coupled_scenes(self):
        config = load_config(); policy = RenderPolicy(depth_variance=1e-6); device = 'cpu'
        ds, p, c, r, parts, cams, q = setup(device, frames=3, width=40, height=30)
        Js = {k: joint_transform(MODEL, float(q[k])) for k in (1, 2)}
        obs = {k: FrameObservation([ds.observation(k, ci) for ci in range(len(cams))], config, ObservationPolicy(), device) for k in Js}
        core = torch.as_tensor(parts == 1)
        rows = []
        for seed in range(3):
            g = np.random.default_rng(seed)
            # coupled candidates: neighbours around one moving point and one static point at the part boundary
            centre = p[g.choice(np.flatnonzero(parts == 1))]
            near = torch.argsort((p-centre).norm(dim=1))[:10]
            cand = torch.zeros(len(p), dtype=torch.bool); cand[near] = True
            fixed = core & ~cand
            best, best_z = float('inf'), None
            for bits in itertools.product([False, True], repeat=len(near)):
                mv = fixed.clone(); mv[near] = torch.tensor(bits)
                E = total_energy(obs, cams, p, mv, Js, c, r, policy)
                if E < best: best, best_z = E, mv.clone()
            mv, E, hist, conv = block_icm(obs, cams, p, fixed, Js, c, r, policy, blocks=2, cycles=6, cand=cand)
            rows.append({'seed': seed, 'exact_map_energy': best, 'icm_energy': E, 'same_labels': bool(torch.equal(mv[near], best_z[near]))})
            self.assertLessEqual(E, best+1e-9*abs(best), rows[-1])
        print(rows)


class SupportLabelsTest(unittest.TestCase):
    def test_labels_on_textured_scene(self):
        from estimator.phase1 import endpoint_catalogue
        from estimator.sparse_iekf import IEKFConfig
        from estimator.point_renderer import footprints
        from joint_reduction import support as image_support
        from utils.backend import Backend
        config = load_config()
        ds = two_plate_scene('prismatic', frames=16, cameras=4)
        _, part, _ = endpoint_catalogue(ds, 'first', config, IEKFConfig()); rho = footprints(part, ds)
        parts = source_parts(ds, part.source)
        core = parts == 1
        core[np.flatnonzero(core)[::3]] = False                   # a third of the true movers left uncertain
        sel = {**MODEL, 'q': (np.linalg.norm([1., .3, 0.])*.02*np.arange(16)).tolist()}
        policy = RenderPolicy(depth_variance=config.depth_variance)
        out = image_support.run(ds, config, Backend(DEVICES[-1]), part, rho, core, sel, 0, tempfile.mkdtemp(), policy, ObservationPolicy(), cycles=6)
        lab = out['labels']; left = (parts == 1) & ~core
        moving = lab == 'MOVING_JOINT_COMPATIBLE'; static = lab == 'STATIC_COMPATIBLE'
        print(out['counts'], 'moving precision', (parts[moving] == 1).mean() if moving.any() else None,
              'static precision', (parts[static] == 0).mean() if static.any() else None, 'recall of uncertain movers', moving[left].mean())
        self.assertGreaterEqual((parts[static] == 0).mean(), .95)
        self.assertGreaterEqual((parts[moving] == 1).mean(), .95)     # SUPPORT_VALIDATED gate (sign rule measured 0.81: disocclusion hole filling)
        self.assertGreater(moving[left].mean(), .3)                    # some of the uncertain movers are recovered


class TexturelessSlideTest(unittest.TestCase):
    def test_textureless_plate_sliding_in_plane_stays_ambiguous(self):
        """Interior points of a uniform plate sliding in its own plane are observationally indistinguishable."""
        from estimator.phase1 import endpoint_catalogue
        from estimator.sparse_iekf import IEKFConfig
        from estimator.point_renderer import footprints
        from joint_reduction import support as image_support
        from utils.backend import Backend
        import tests.image_fixtures as fx
        config = load_config()
        ds = two_plate_scene('prismatic', frames=10, cameras=4)
        orig = fx.texture
        fx.texture = lambda s_, t_, phase=0.: np.broadcast_to(np.array([.6, .4, .3]), (*np.shape(s_), 3)).copy() if phase else orig(s_, t_, phase)
        try:
            ds._cache.clear()
            _, part, _ = endpoint_catalogue(ds, 'first', config, IEKFConfig()); rho = footprints(part, ds)
            parts = source_parts(ds, part.source)
            plate = part.xyz[:, 2] > .03
            interior = plate & (np.abs(part.xyz[:, 0]) < .12) & (np.abs(part.xyz[:, 1]) < .06)
            core = plate & ~interior
            sel = {**MODEL, 'q': (np.linalg.norm([1., .3, 0.])*.02*np.arange(10)/4).tolist()}   # small in-plane slide
            out = image_support.run(ds, config, Backend(DEVICES[-1]), part, rho, core, sel, 0, tempfile.mkdtemp(), RenderPolicy(depth_variance=config.depth_variance), ObservationPolicy())
        finally:
            fx.texture = orig
        lab = out['labels'][interior]
        print('textureless interior labels:', {l: int((lab == l).sum()) for l in set(lab)})
        self.assertLess((lab == 'MOVING_JOINT_COMPATIBLE').mean(), .05)
        self.assertLess((lab == 'STATIC_COMPATIBLE').mean(), .05)


class BidirectionalCompareTest(unittest.TestCase):
    def test_world_comparison_sign_and_gauge(self):
        q = np.linspace(0, .5, 11)
        f = {'type': 'PRISMATIC', 'axis': [0, -1, 0], 'q': q.tolist()}
        b = {'type': 'PRISMATIC', 'axis': [0, 1, 0], 'q': (-(q-q[-1])).tolist()}   # opposite sign, anchored at the end
        out = compare(f, b, .005, 1.)
        self.assertEqual(out['axis_sign'], -1.); self.assertLess(out['axis_angle_rad'], 1e-12)
        self.assertLess(out['q_rmse'], 1e-12)
        bad = compare(f, {**b, 'axis': [1, 0, 0]}, .005, 1.)
        self.assertGreater(bad['axis_angle_rad'], 1.5)


if __name__ == '__main__':
    unittest.main()
