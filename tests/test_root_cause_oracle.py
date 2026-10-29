"""The root-cause oracle (tools/root_cause_oracle.py, independent NumPy) reproduces the production free IEKF of a traced
point: every iterate's A, b, delta, the accepted state and P+, under the original and the current policies; its exact
step attribution sums to delta; the velocity correction is the covariance coupling of the position correction."""
import unittest
import numpy as np
import torch
from tests.projective_fixtures import noise, POLICY
from tests.image_fixtures import two_plate_scene, catalog, look_at
from estimator.projective_camera import CameraRig
from estimator.projective_sensor import MeasuredFrame
from estimator.point_kalman_cuda import update_points
from tools import root_cause_oracle as O

ORIGINAL = {**POLICY, 'update_acceptance': 'final_iterate', 'row_innovation_gate': 'depth_front_only', 'final_visibility': 'frozen'}


def scene():
    ds = two_plate_scene('static', frames=1, cameras=4)
    ds.cameras = [look_at(c.name, -c.world_to_camera[:3, :3].T@c.world_to_camera[:3, 3], [0, 0, 0], 320, 240, 400.) for c in ds.cameras]
    xyz, rgb, _, _ = catalog(ds, 0)
    ok = (np.abs(xyz[:, 0]) < .41) & (np.abs(xyz[:, 1]) < .26) & (np.abs(xyz[:, 2]) < 1e-9) & ((np.abs(xyz[:, 0]) > .34) | (np.abs(xyz[:, 1]) > .23))
    pick = np.sort(np.random.default_rng(0).choice(np.flatnonzero(ok), 30, replace=False))
    return ds, xyz[pick], rgb[pick]


class OracleReplayTest(unittest.TestCase):
    def check(self, policy, offset, sig=.02, ids=(3, 7)):
        ds, x, rgb = scene()
        rig, frame = CameraRig(ds.cameras, 'cpu'), MeasuredFrame(ds, 0, 'cpu', torch.float64)
        n = len(x)
        s = torch.zeros((n, 6), dtype=torch.float64); s[:, :3] = torch.as_tensor(x)+torch.as_tensor(offset)
        P = torch.zeros((n, 6, 6), dtype=torch.float64); P[:, :3, :3] = sig**2*torch.eye(3, dtype=torch.float64); P[:, 3:, 3:] = torch.eye(3, dtype=torch.float64)
        P[:, :3, 3:] = P[:, 3:, :3] = .5*sig*torch.eye(3, dtype=torch.float64)/30          # prior position-velocity coupling, as after a prediction
        nz = noise()
        sp, Pp, rep, tr = update_points(s, P, frame, rig, torch.as_tensor(rgb), nz, policy, 10, 1e-8, trace_ids=list(ids))
        cams = [O.Cam(c) for c in ds.cameras]; imgs = [O.Image(ds.observation(0, c), np.float64) for c in range(len(cams))]
        on = O.Noise(nz.rgb_variance.numpy(), nz.depth_variance, nz.depth_squared_variance, nz.rgb_model_variance, nz.depth_model_variance, nz.interpolation_variance)
        for pid in ids:
            fin = tr['final'][pid]
            frozen = [O.CODE[c['visibility_frozen']] for c in tr['iterations'][0]['points'][pid]['cameras']]
            recheck = [c['visibility_fresh'] == 'SELF_OCCLUDED' for c in fin['cameras_final_iterate']] if policy['final_visibility'] == 'recheck_self_occlusion' else None
            prob = O.Problem(cams, imgs, rgb[pid], np.array(fin['s_minus']), np.array(fin['P_minus']), frozen, on, policy, 10, 1e-8, recheck)
            res = prob.run(policy['update_acceptance'])
            for j, h in enumerate(res['history']):
                t = tr['iterations'][j]['points'][pid]
                np.testing.assert_allclose(h['s'], t['state'], rtol=0, atol=1e-12)
                np.testing.assert_allclose(h['A'], t['A'], rtol=1e-10, atol=1e-6)
                np.testing.assert_allclose(h['delta'], t['delta'], rtol=1e-9, atol=1e-13)
                att = prob.attribution(h['s'], h['rows'], h['A'])
                np.testing.assert_allclose(sum(np.asarray(v['delta']) for v in att.values()), h['delta'], rtol=1e-9, atol=1e-14)
            self.assertEqual(res['observed'], bool(rep['observed'][pid]))
            np.testing.assert_allclose(res['s_plus'], fin['s_plus'], rtol=0, atol=1e-12)
            np.testing.assert_allclose(res['P_plus'], fin['P_plus'], rtol=1e-9, atol=1e-16)
            d = res['s_plus']-prob.s_minus
            np.testing.assert_allclose(d[3:], prob.P_minus[3:, :3]@np.linalg.solve(prob.P_minus[:3, :3], d[:3]), rtol=1e-8, atol=1e-13)
            if res['observed']:
                j = prob.joseph(res['rows_accepted'])
                self.assertLess(j['max_abs_difference'], 1e-12)
            xf = res['s_final_iterate'][:3]                               # the final probe is taken at the final iterate
            for c in range(len(cams)):
                p = fin['cameras_final_iterate'][c]
                o = O.row(imgs[c], cams[c], xf, rgb[pid], on, policy, prob.P_minus[:3, :3], xf-prob.s_minus[:3])
                self.assertEqual(O.NAMES[o['reason']], p['reason_if_sampled'])
                if 'H' in o: np.testing.assert_allclose(o['H'], p['H_x'], rtol=1e-9, atol=1e-9)

    def test_original_policy(self):
        self.check(ORIGINAL, [.0015, -.001, .004])

    def test_current_policy(self):
        self.check(POLICY, [.0015, -.001, .004])

    def test_rejected_large_offset_current_policy(self):
        self.check({**POLICY, 'row_innovation_gate': 'depth_front_only'}, [.05, 0., 0.], sig=.004)


if __name__ == '__main__':
    unittest.main()
