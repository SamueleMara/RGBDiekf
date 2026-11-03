"""Actual local-correlated estimator on controlled motion (spec tests 4, 8, 22): not a forward-warp check.
All numbers specify this toy sensor only; real-scene motion is evaluated post hoc against GT, never here."""
import unittest
import numpy as np
import torch
from tests.test_temporal_warp import rig, frame
from estimator.projective_sensor import SensorNoise
from estimator.point_kalman_cuda import predict
from estimator.temporal_reference import ReferenceTable
from estimator.temporal_zbuffer import first_hit_pixel_owner
from estimator.temporal_rgbd_terms import build_temporal_rows
from estimator.temporal_simplified_iekf import update_points_temporal_simplified
from estimator.temporal_local_covariance import KEY
from estimator.temporal_local_iekf import current_marginals
from estimator.temporal_local_run import LocalFilter

DEVICES = ['cpu']+(['cuda'] if torch.cuda.is_available() else [])
POLICY = {'same_surface_n_sigma': 3., 'same_surface_max_incidence_deg': 80.}
LOCAL_MOTION_WITNESSES = []


def scene(device, dx_per_frame, frames=2):
    """Camera 0 sees a translating textured plane at z=2 (fx=4 -> 2*dx pixels per frame); camera 1 is masked out."""
    out = []
    for k in range(frames):
        f = frame(device)
        f.k = k
        f.mask[1] = False
        f.depth[:] = 2.
        f.rgb[0, :, :, 0] -= .02*2*dx_per_frame*k
        f.rgb[0, :, :, 1] -= .03*2*dx_per_frame*k
        out.append(f)
    return out


def local_filter(device, f0, lag=3):
    noise = SensorNoise(torch.ones(3, dtype=torch.float64, device=device)*1e-8, 1e-8, 0.)
    warp = np.eye(4)*1e-8
    filt = LocalFilter(rig(device), noise, POLICY, warp, np.eye(3)*.01, lag, 10**8, 8, f0, np.array([[0, 0, 27]]), np.array([[0., 0., 2.]]))
    filt.initialize(f0, np.eye(2)/12, np.eye(3))
    return filt, noise, warp


class LocalMotionTests(unittest.TestCase):
    def test_cell_boundary_motion_both_directions(self):
        dt = .04
        for dx in (.1, -.1):
            for device in DEVICES:
                with self.subTest(dx=dx, device=device):
                    f0, f1 = scene(device, dx)
                    filt, _, _ = local_filter(device, f0)
                    step = filt.step({0: f0, 1: f1}, 1, 1, dt, 10, 1e-8, [0])
                    self.assertTrue(bool(step['update'].assimilated[0]))
                    tr = step['post']['traces'][0]
                    blocks = {b['kind']: b['columns'] for b in tr['posterior_layout']}
                    mean = np.asarray(tr['mean_plus'])
                    a, b = blocks['source_chart']
                    src = [blk['columns'] for blk in tr['posterior_layout'] if blk['kind'] == 'raw_sensor_error' and blk['frame'] == 0 and blk['pixel'] == 27][0]
                    anchor = filt.chart.evaluate(torch.tensor([0], device=device), torch.tensor(mean[a:b][None], device=device),
                                                 torch.tensor([mean[src[0]+3]], device=device))[0][0].cpu().numpy()
                    relative = float(step['s'][0, 0].cpu())-anchor[0]
                    velocity = float(step['s'][0, 3].cpu())
                    self.assertLess(abs(relative-dx), .002)
                    self.assertLess(abs(velocity-dx/dt), .08)
                    LOCAL_MOTION_WITNESSES.append({'device': device, 'true_dx_m': dx, 'relative_dx_m': relative, 'velocity_m_s': velocity,
                                                   'true_velocity_m_s': dx/dt, 'fresh_raw_keys': step['solver']['counters']['fresh_raw_keys_registered'],
                                                   'source_key_reuse_rows': step['lr'].reused_source_keys})

    def test_static_equilibrium_has_no_drift(self):
        for device in DEVICES:
            frames = scene(device, 0., 4)
            filt, _, _ = local_filter(device, frames[0])
            s0 = None
            for k in range(1, 4):
                step = filt.step({j: frames[j] for j in range(max(0, k-2), k+1)}, k, k, .04, 10, 1e-8)
                if s0 is None:
                    s0 = step['s'][0].clone()
            self.assertLess(float((step['s'][0, :3]-s0[:3]).norm()), 1e-6)
            self.assertLess(float(step['s'][0, 3:].norm()), 1e-3)
            LOCAL_MOTION_WITNESSES.append({'device': device, 'static_speed_after_3_frames_m_s': float(step['s'][0, 3:].norm())})

    def test_state_noise_correlation_persists_and_differs_from_simplified(self):
        dx, dt = .05, .04
        f0, f1, f2 = scene('cpu', dx, 3)
        filt, noise, warp = local_filter('cpu', f0)
        filt.step({0: f0, 1: f1}, 1, 1, dt, 10, 1e-8)
        L = filt.bank.layout
        keys = torch.where((L.point == 0) & (L.kind == KEY))[0]
        self.assertTrue(len(keys))                                   # the frame-1 current raw error is retained ...
        self.assertTrue(all(int(L.frame[j]) == 1 for j in keys))     # ... as a key of camera 0's new reference frame
        mean, cov = filt.bank.gather(torch.tensor([0]))
        start = int(L.start[keys[0]])
        cross_post = float(cov[0, :6, start:start+4].abs().max())
        self.assertGreater(cross_post, 0.)
        step = filt.step({0: f0, 1: f1, 2: f2}, 2, 2, dt, 10, 1e-8, [0])
        tr = step['post']['traces'][0]
        reused = [b for b in tr['posterior_layout'] if b['kind'] == 'raw_sensor_error' and b['frame'] == 1]
        self.assertTrue(reused)
        self.assertGreater(step['lr'].reused_history_keys, 0)
        # simplified ablation from the same 6x6 marginal prior: fixed reference, independent noise copies
        prior_s, prior_P = mean[:, :6], cov[:, :6, :6]
        m, P = predict(prior_s, prior_P, dt, torch.eye(3, dtype=torch.float64)*.01)
        r = rig()
        refs = ReferenceTable(2, 1, 'cpu', torch.float64)
        refs.frame[0, 0], refs.ordinal[0, 0], refs.state[0, 0] = 1, 1, prior_s[0]
        noise_t = noise

        def build(x):
            own = first_hit_pixel_owner(x[:, :3], r, f2)
            rows, reason, life = build_temporal_rows(x, own, f2, refs, {1: f1, 2: f2}, r, noise_t, POLICY, torch.as_tensor(warp), 2, 3)
            return rows, own, reason, life
        simple = update_points_temporal_simplified(m, P, build, f2, 10, 1e-8)
        dP = float((step['P'][0]-simple[1][0]).abs().max()/simple[1][0].abs().max())
        self.assertGreater(dP, 1e-3)
        LOCAL_MOTION_WITNESSES.append({'test': 'persistence', 'Cov_state_retained_raw_error_max_abs_after_frame1': cross_post,
                                       'reused_history_keys_frame2': step['lr'].reused_history_keys,
                                       'local_vs_simplified_covariance_relative_difference': dP,
                                       'local_dx_frame2_m': float(step['s'][0, 0]), 'simplified_dx_frame2_m': float(simple[0][0, 0])})
