"""Contract section 7 gates: information-supported coherence vs the pinned extractor and a zero-order hold."""
import json, unittest
from pathlib import Path
import numpy as np
import torch
from estimator.motion_coherence_support import canonical_knn_graph, build_supernodes
from tests.reference_motion import edge_strain
from rigeuler_ref import coherence_extraction as pinned
from estimator import supported_coherence as sc

REPORT = Path(__file__).resolve().parents[1]/'reports/coherence_occlusion_simulation.json'


def rot_z(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.]])


def scene(K=30, seed=0, motion='translation'):
    """Nodes on a static plate plus a moving plate; returns true node trajectories [K,n,3] and the moving mask."""
    g = np.random.default_rng(seed)
    base = np.c_[g.uniform(-.5, .5, (150, 2)), np.zeros(150)]
    part = np.c_[g.uniform(.1, .4, (40, 2)), np.full(40, .05)]
    X0 = np.concatenate([base, part]); moving = np.r_[np.zeros(150, bool), np.ones(40, bool)]
    traj = np.repeat(X0[None], K, 0)
    for k in range(K):
        if motion == 'translation': traj[k, moving] = part+np.array([.01, .004, 0])*k
        else: traj[k, moving] = (part-[.1, .1, .05])@rot_z(.03*k).T+[.1, .1, .05]
    return traj+g.normal(0, 2e-4, traj.shape), moving


def run(traj, observed, method):
    """'reference' (pinned, raw), 'zoh' (hold at last supported frame), 'supported' (this adaptation)."""
    K, n = traj.shape[:2]
    X = torch.as_tensor(traj)
    if method == 'zoh':
        X = X.clone()
        for k in range(1, K):
            X[k] = torch.where(torch.as_tensor(observed[k])[:, None], X[k], X[k-1])
    ei, ej = canonical_knn_graph(X[0], 12)
    if method == 'supported':
        T, om, ids, _ = sc.supported_node_trajectories(X, observed, torch.arange(n), 0)
        st = sc.weighted_edge_strain(T, om, ei, ej)
        keep = st['supported_edges']
        res = sc.extract_moving_component(T, om, ei[keep], ej[keep], st['C'][keep], n)
    else:
        C = edge_strain(X, ei, ej)['C']
        res = pinned.extract_moving_component(X, ei, ej, C, n, verbose=False)
    sel = np.zeros(n, bool); sel[res['moving_nodes'].numpy()] = True
    return sel, res


def iou(a, b):
    return float((a & b).sum()/max((a | b).sum(), 1))


class SupportedCoherenceTest(unittest.TestCase):
    def test_reduces_to_pinned_extractor_with_full_support(self):
        for motion in ('translation', 'rotation'):
            traj, moving = scene(motion=motion)
            obs = np.ones(traj.shape[:2], bool)
            a, ra = run(traj, obs, 'reference'); b, rb = run(traj, obs, 'supported')
            self.assertTrue(np.array_equal(a, b), motion)
            self.assertEqual(ra['labels'], rb['labels'])
            for ca, cb in zip(ra['candidates'], rb['candidates']):
                for key in ('eta_cv', 'delta_bic', 'delta_rss_cv'):
                    x, y = ca['rigidity'].get(key), cb['rigidity'].get(key)
                    if x is not None and np.isfinite(x): self.assertLess(abs(x-y), 1e-9*max(1., abs(x)), key)
            self.assertGreater(iou(b, moving), .9)

    def test_node_trajectory_and_strain_reduce_exactly(self):
        traj, _ = scene()
        g = np.random.default_rng(1)
        pts = np.repeat(traj, 4, axis=1)+g.normal(0, 1e-3, (traj.shape[0], 4*traj.shape[1], 3))
        sid = torch.arange(traj.shape[1]).repeat_interleave(4)
        T, om, ids, _ = sc.supported_node_trajectories(pts, np.ones(pts.shape[:2], bool), sid, 0)
        from tests.reference_motion import supernode_trajectories
        ref = supernode_trajectories(pts, sid)['trajectories']
        self.assertLess(float((T-ref).abs().max()), 1e-12)
        ei, ej = canonical_knn_graph(T[0], 12)
        a, b = edge_strain(T, ei, ej)['C'], sc.weighted_edge_strain(T, om, ei, ej)['C']
        self.assertLess(float((a-b).abs().max()), 1e-12)

    def test_simulated_occlusion_against_reference_and_zero_order_hold(self):
        out = {}
        K = 30
        # A: an occluded STATIC patch whose Phase-1 estimate drifts at constant velocity (prediction only).
        traj, moving = scene(K)
        est = traj.copy(); obs = np.ones(traj.shape[:2], bool)
        patch = np.flatnonzero(~moving)[np.argsort(traj[0, ~moving, 0])[:30]]
        for k in range(10, K):
            obs[k, patch] = False; est[k, patch] = traj[k, patch]+np.array([.012, -.006, 0])*(k-9)
        out['A_static_patch_drifting_while_occluded'] = {m: iou(run(est, obs, m)[0], moving) for m in ('reference', 'zoh', 'supported')}
        # B: half of the MOVING part hidden from frame 15 while it reverses; its estimate keeps the old velocity.
        traj, moving = scene(K)
        mv = np.flatnonzero(moving)
        for k in range(15, K): traj[k, mv] = traj[14, mv]-(traj[k, mv]-traj[14, mv])
        est = traj.copy(); obs = np.ones(traj.shape[:2], bool)
        hidden = mv[np.argsort(traj[0, mv, 0])[:20]]
        for k in range(15, K):
            obs[k, hidden] = False; est[k, hidden] = traj[14, hidden]+np.array([.01, .004, 0])*(k-14)
        out['B_moving_half_hidden_and_reversing'] = {m: iou(run(est, obs, m)[0], moving) for m in ('reference', 'zoh', 'supported')}
        # C: a moving part hidden ENTIRELY for the second half (true motion continues): no evidence then, still found from the rest.
        traj, moving = scene(K, motion='rotation')
        est = traj.copy(); obs = np.ones(traj.shape[:2], bool)
        for k in range(15, K):
            obs[k, mv] = False; est[k, mv] = traj[14, mv]
        out['C_moving_part_fully_hidden_second_half'] = {m: iou(run(est, obs, m)[0], moving) for m in ('reference', 'zoh', 'supported')}
        # D: a STATIC scene whose occluded patch drifts (prediction only): any selection is a false moving body.
        traj, _ = scene(K, motion='translation'); traj[:, 150:] = traj[0, 150:]
        est = traj.copy(); obs = np.ones(traj.shape[:2], bool)
        for k in range(10, K):
            obs[k, patch] = False; est[k, patch] = traj[k, patch]+np.array([.012, -.006, 0])*(k-9)
        out['D_static_scene_patch_drifting_while_occluded'] = {m: 1.-float(run(est, obs, m)[0].mean()) for m in ('reference', 'zoh', 'supported')}
        # E: half of the moving part hidden from frame 15 while its motion CONTINUES (the prediction is right).
        traj, moving = scene(K)
        est = traj.copy(); obs = np.ones(traj.shape[:2], bool)
        for k in range(15, K): obs[k, hidden] = False
        out['E_moving_half_hidden_motion_continues'] = {m: iou(run(est, obs, m)[0], moving) for m in ('reference', 'zoh', 'supported')}
        REPORT.parent.mkdir(exist_ok=True)
        REPORT.write_text(json.dumps({'metric': 'IoU of the selected moving nodes with the true moving set (D: 1 - selected fraction, static truth)',
                                      'scenarios': out}, indent=1))
        print(json.dumps(out, indent=1))
        for name, r in out.items():
            if name.startswith('D_'): continue     # documented property of the pinned extractor, asserted separately
            self.assertGreater(r['supported'], .9, name)

    def test_pinned_extractor_resolves_a_component_on_static_noise(self):
        """Documented limitation (not an adaptation artefact): the pinned selection resolves a 'moving' component on a
        static, noise-only scene, so static scenes must be rejected downstream (joint predictive comparison)."""
        traj, _ = scene(30); traj[:, 150:] = traj[0, 150:]+np.random.default_rng(5).normal(0, 2e-4, traj[:, 150:].shape)
        obs = np.ones(traj.shape[:2], bool)
        a, ra = run(traj, obs, 'reference'); b, rb = run(traj, obs, 'supported')
        self.assertEqual(ra['status'], 'RESOLVED'); self.assertTrue(np.array_equal(a, b))

    def test_unsupported_edges_are_removed_not_zero_strain(self):
        traj, _ = scene()
        obs = np.ones(traj.shape[:2], bool); obs[1:, :5] = False
        T, om, _, _ = sc.supported_node_trajectories(traj, obs, torch.arange(traj.shape[1]), 0)
        ei, ej = canonical_knn_graph(T[0], 12)
        st = sc.weighted_edge_strain(T, om, ei, ej)
        hidden = (ei < 5) | (ej < 5)                                  # only the gauge frame: zero strain by construction
        self.assertTrue(bool((~st['supported_edges'][hidden]).all()))
        self.assertTrue(bool(st['supported_edges'][~hidden].all()))


class CoherenceSceneGates(unittest.TestCase):
    def test_pure_global_drift(self):
        """Documented limitation of the PINNED selection (reproduced, not introduced): under a pure global drift it
        qualifies a candidate whose motion relative to its complement is at the noise level, and the world-frame core
        refinement then recruits every node. Downstream stages must not read this as articulation."""
        traj, _ = scene(30); traj[:, 150:] = traj[0, 150:]
        traj = traj+np.array([.004, -.002, .001])*np.arange(30)[:, None, None]          # the whole scene drifts rigidly
        obs = np.ones(traj.shape[:2], bool)
        a, ra = run(traj, obs, 'reference'); b, rb = run(traj, obs, 'supported')
        self.assertTrue(np.array_equal(a, b))
        c = next(c for c in rb['candidates'] if c['candidate_id'] == rb['selection']['selected'])
        print('global drift:', rb['status'], int(b.sum()), 'selected relative motion ratio', c['rigidity']['relative_motion_ratio'])
        self.assertLess(c['rigidity']['relative_motion_ratio'], .01)

    def test_nonrigid_random_drift_selects_no_coherent_body_of_size(self):
        g = np.random.default_rng(3)
        traj, _ = scene(30); traj[:, 150:] = traj[0, 150:]
        traj = traj+np.cumsum(g.normal(0, 2e-3, traj.shape), axis=0)   # independent random walks: no rigid body
        sel, res = run(traj, np.ones(traj.shape[:2], bool), 'supported')
        etas = [c['rigidity'].get('eta_cv') for c in res['candidates'] if not c['rigidity'].get('degenerate')]
        print('random drift:', res['status'], int(sel.sum()), 'max held-out eta', max(etas))
        if res['status'] == 'RESOLVED':
            chosen = next(c for c in res['candidates'] if c['candidate_id'] == res['selection']['selected'])
            self.assertLess(chosen['rigidity']['eta_cv'], .5)          # no common rigid trajectory predicts held-out nodes well

    def test_reverse_processing_order_selects_the_same_part(self):
        traj, moving = scene(30, motion='rotation')
        obs = np.ones(traj.shape[:2], bool)
        fwd, _ = run(traj, obs, 'supported'); bwd, _ = run(traj[::-1].copy(), obs, 'supported')
        self.assertTrue(np.array_equal(fwd, bwd)); self.assertGreater(iou(fwd, moving), .9)

    def test_independent_of_phase1_group_occupancy(self):
        import tempfile
        from types import SimpleNamespace
        from estimator.motion_coherence_support import run as coherence_run
        traj, moving = scene(12, motion='rotation')
        pts = np.repeat(traj, 5, axis=1)+np.random.default_rng(2).normal(0, 1e-4, (12, 5*traj.shape[1], 3))
        arrays = {'xyz': pts, 'point_observed': np.ones(pts.shape[:2], bool)}
        out = []
        for groups in (8, 64):
            part = SimpleNamespace(xyz=pts[0], count=groups, membership=np.arange(len(pts[0])) % groups)
            r = coherence_run(part, arrays, np.arange(12)/30, tempfile.mkdtemp(), 0, count=96)
            out.append(r['point_support'])
        self.assertTrue(np.array_equal(out[0], out[1]))


if __name__ == '__main__':
    unittest.main()
