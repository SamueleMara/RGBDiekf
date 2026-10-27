"""Step-1 coherence parity gates (CLAUDE_STEP1_RIGEULER_COHERENCE_SUPERNODE_PARITY_AUDIT.md).

The pinned RigEuler3DGS 481b2cf2 sources are loaded from the Git blobs (tools/audit_coherence_parity.load_pinned).
Saved-run gates are skipped when the run or the reference repository is not available on this machine.
"""
import tempfile, unittest
from pathlib import Path
import numpy as np
import torch
from tools.audit_coherence_parity import load_pinned, run_cell, load_run, COMMIT, PINNED

REF = Path(__file__).resolve().parents[2]/'RigEuler3DGS'
V2 = Path('/media/samuelemara/Extreme SSD/Samuele/results_11_10_26/RGBDekf/image_iekf_v2_rigorous_20261024')
V1 = Path('/media/samuelemara/Extreme SSD/Samuele/results_11_10_26/RGBDekf/image_iekf_final_20261024')


@unittest.skipUnless((REF/'.git').exists(), 'pinned reference repository not available')
class CoherenceParityTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_default_dtype(torch.float64)
        cls.mods, cls.blobs = load_pinned(REF, tempfile.mkdtemp())

    def test_pinned_blobs_loaded(self):
        self.assertEqual(set(self.blobs), set(PINNED))
        self.assertEqual(self.blobs['estimator/coherence_extraction.py']['sha256'][:16], '5931ee32ca681fb8')

    def test_toy_two_body_identical_and_invariant(self):
        mcs, sn = self.mods['motion_coherence_support'], self.mods['supernodes']
        g = torch.Generator().manual_seed(0)
        base = torch.cat([torch.rand(600, 2, generator=g)-.5, torch.zeros(600, 1)], 1)
        part = torch.cat([torch.rand(200, 2, generator=g)*.3+.1, torch.full((200, 1), .05)], 1)
        X = torch.stack([torch.cat([base, part+torch.tensor([.01, .004, 0.])*k]) for k in range(30)])+torch.randn(30, 800, 3, generator=g)*2e-4
        sid = sn.build_supernodes(X[0].float(), num_supernodes=40, seed=0).supernode_id.long()
        s = mcs.supernode_trajectories([X[k] for k in range(30)], sid); Xn = s['trajectories'][:, s['valid'], :]
        gr = mcs.canonical_knn_graph(Xn[0], k=12)
        out = {e: run_cell(e, Xn, torch.ones(Xn.shape[:2]), gr['edge_i'].long(), gr['edge_j'].long(), self.mods, e, True) for e in ('pinned', 'local')}
        self.assertEqual(out['pinned']['labels'], out['local']['labels'])
        frac = torch.stack([(sid == v)[600:].sum()/(sid == v).sum() for v in range(Xn.shape[1])])
        sel = set(out['pinned']['final_moving_nodes'])
        self.assertTrue({int(v) for v in torch.nonzero(frac > .5).reshape(-1)} <= sel)          # every part-majority node
        self.assertTrue(all(float(frac[v]) > 0 for v in sel))                                   # no purely static node
        self.assertTrue(out['pinned']['refinement']['recruits_explained_by_EM_lt_EI'])

    @unittest.skipUnless((V2/'first/phase1/state.npy').exists(), 'saved v2 forward run not available')
    def test_saved_v2_bundle_local_equals_pinned_and_reproduces_failure(self):
        mcs, sn = self.mods['motion_coherence_support'], self.mods['supernodes']
        v2 = load_run(V2); X = torch.as_tensor(v2['xyz']); K = X.shape[0]
        sid = sn.build_supernodes(torch.as_tensor(v2['canonical']).float(), num_supernodes=192, seed=0).supernode_id.long()
        self.assertTrue(torch.equal(sid, torch.as_tensor(v2['reps']['membership'])))             # v2 used the pinned partition
        s = mcs.supernode_trajectories([X[k] for k in range(K)], sid); Xn = s['trajectories'][:, s['valid'], :]
        gr = mcs.canonical_knn_graph(Xn[0], k=12); ei, ej = gr['edge_i'].long(), gr['edge_j'].long()
        ones = torch.ones(Xn.shape[:2])
        for refine in (False, True):
            p = run_cell('p', Xn, ones, ei, ej, self.mods, 'pinned', refine); l = run_cell('l', Xn, ones, ei, ej, self.mods, 'local', refine)
            self.assertEqual(p['labels'], l['labels']); self.assertEqual(p['seed_nodes'], l['seed_nodes'])
        # documented failure of the PINNED code on these inputs: seed 183 (refine=False) -> 192/192 (refine=True)
        self.assertEqual(len(p['seed_nodes']), 183); self.assertEqual(len(p['final_moving_nodes']), 192)
        # every high-world-motion candidate in the pool has < 5 nodes and is therefore degenerate under 5-fold CV
        fast = [c for c in p['candidates'] if (c['world_delta_rss_cv_per_node_frame'] or 0) > 1e-3]
        self.assertTrue(all(c['num_nodes'] < 5 or c['num_boundary_edges'] == 0 or c['num_nodes'] > 150 for c in fast))

    @unittest.skipUnless((V1/'first/phase1/state.npy').exists() and (V2/'first/phase1/state.npy').exists(), 'saved runs not available')
    def test_failure_follows_the_input_trajectories(self):
        mcs, sn = self.mods['motion_coherence_support'], self.mods['supernodes']
        res = {}
        for tag, run in (('v1', V1), ('v2', V2)):
            v = load_run(run); X = torch.as_tensor(v['xyz']); K = X.shape[0]
            sid = sn.build_supernodes(torch.as_tensor(v['canonical']).float(), num_supernodes=192, seed=0).supernode_id.long()
            s = mcs.supernode_trajectories([X[k] for k in range(K)], sid); Xn = s['trajectories'][:, s['valid'], :]
            gr = mcs.canonical_knn_graph(Xn[0], k=12)
            res[tag] = run_cell(tag, Xn, torch.ones(Xn.shape[:2]), gr['edge_i'].long(), gr['edge_j'].long(), self.mods, 'pinned', True)
        self.assertLess(len(res['v1']['final_moving_nodes']), 40)          # identical pinned pipeline, earlier trajectories: a proper part
        self.assertEqual(len(res['v2']['final_moving_nodes']), 192)        # same pipeline, v2 trajectories: the whole scene


if __name__ == '__main__':
    unittest.main()
