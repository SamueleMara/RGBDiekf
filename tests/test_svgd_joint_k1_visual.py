"""Display-only viewer of the joint K = 1 SVGD run (debug/svgd_joint_k1_debug.py) + its headless-Chrome QA (tools/qa_svgd_joint_k1_debug.py).

    CUDA_VISIBLE_DEVICES= TMPDIR=/tmp/claude-1000 K1_VIZ_TMP=<scratch dir> python -B -m unittest tests.test_svgd_joint_k1_visual -v

A tiny CPU synthetic run (4 frames, 3 SVGD particles x 2 iterations, 2 MAP iterations; ~10-30 s) is made once with the unchanged runner
tools/run_svgd_joint_k1_synthetic.py; K1_VIZ_ORACLE=1 additionally runs the real post-hoc oracle (slow on CPU). Checks: arrays decode to the
saved values at the display IDs, one deterministic display subset, observed clouds rebuilt with the runner's loader (sha256 == loader_counts),
nothing outside RUN/debug_k1 changes (every other file hashed before / after), post-hoc GT layer only when an oracle folder exists, browser QA
PASS when Chrome is available (else skipTest with the reason). Temporary runs are left in K1_VIZ_TMP (nothing is deleted).
"""
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from tools.qa_svgd_support_debug import read_js, dec

CHROME = '/usr/bin/google-chrome'
ENV = {**os.environ, 'CUDA_VISIBLE_DEVICES': '', 'OMP_NUM_THREADS': os.environ.get('OMP_NUM_THREADS', '8')}


def tree_hashes(root, exclude='debug_k1'):
    root = Path(root); return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                               for p in sorted(root.rglob('*')) if p.is_file() and p.relative_to(root).parts[0] != exclude}


class K1Visual(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        base = os.environ.get('K1_VIZ_TMP'); Path(base).mkdir(parents=True, exist_ok=True) if base else None
        cls.tmp = Path(tempfile.mkdtemp(prefix='k1vis_', dir=base)); cls.rd = cls.tmp/'run'
        cfg = json.loads((REPO/'configs/svgd_joint_k1_overnight.json').read_text())
        cfg['svgd']['value'].update(iterations=2, particles=3); cfg['map']['value']['iterations'] = 2
        (cls.tmp/'cfg.json').write_text(json.dumps(cfg, indent=1))
        p = subprocess.run([sys.executable, '-B', str(REPO/'tools/run_svgd_joint_k1_synthetic.py'), '--motion', 'translate', '--frames', '4', '--device', 'cpu',
                            '--config', str(cls.tmp/'cfg.json'), '--output', str(cls.rd)], cwd=REPO, env=ENV, capture_output=True, text=True, timeout=1800)
        if p.returncode: raise RuntimeError(f'synthetic K1 run failed:\n{p.stdout[-3000:]}\n{p.stderr[-3000:]}')
        if os.environ.get('K1_VIZ_ORACLE') == '1':
            p = subprocess.run([sys.executable, '-B', str(REPO/'tools/audit_svgd_joint_k1_oracles.py'), '--run', str(cls.rd), '--synthetic', '--device', 'cpu'],
                               cwd=REPO, env=ENV, capture_output=True, text=True, timeout=3600)
            if p.returncode: raise RuntimeError(f'oracle failed:\n{p.stdout[-2000:]}\n{p.stderr[-2000:]}')
        cls.before = tree_hashes(cls.rd)
        from debug.svgd_joint_k1_debug import write_debug
        cls.out = write_debug(cls.rd, 'light', display_cap=60000)
        cls.after = tree_hashes(cls.rd)

    def meta(self, run=None): return read_js((run or self.rd)/'debug_k1'/'data'/'k1_meta.js')

    def test_nothing_outside_debug_k1_changes(self):
        self.assertEqual(self.before, self.after, 'write_debug modified or created files outside RUN/debug_k1')
        for f in ('index.html', 'k1_3d.html', 'k1_numerical.html', 'three.min.js', 'THREE_LICENSE.txt', 'projective_common.js', 'projective_theme.css', 'data/k1_meta.js', 'data/k1_numeric.js'):
            self.assertTrue((self.rd/'debug_k1'/f).is_file(), f)

    def test_arrays_decode_to_saved_values(self):
        M = self.meta(); sv = self.rd/'svgd_joint_k1'; ids = dec(M['display_ids'], np.int32).astype(int)
        cat = np.load(sv/'reference_catalogue.npz'); N = len(cat['xyz'])
        self.assertEqual((M['N'], M['N_display']), (N, len(ids))); self.assertTrue(np.all(np.diff(ids) > 0))
        np.testing.assert_array_equal(dec(M['xyz'], np.float32).reshape(-1, 3), cat['xyz'][ids].astype(np.float32))
        for key, f in (('gamma', 'gamma_provisional.npy'), ('map_gamma', 'map_gamma.npy'), ('B', 'log_evidence_ratio.npy'), ('B_std', 'log_evidence_ratio_particle_std.npy'),
                       ('evidence_static', 'evidence_static.npy'), ('evidence_moving', 'evidence_moving_conditional.npy')):
            np.testing.assert_array_equal(dec(M[key], np.float32), np.load(sv/f)[ids].astype(np.float32), key)
        np.testing.assert_array_equal(dec(M['identifiability'], np.uint8), np.load(sv/'identifiability_status.npy')[ids].astype(np.uint8))
        np.testing.assert_array_equal(np.asarray(M['pose_particles']), np.load(sv/'pose_particles.npy'))
        np.testing.assert_array_equal(np.asarray(M['map_pose']), np.load(sv/'map_pose.npy'))
        seeds = np.load(sv/'moving_seed_indices.npy'); self.assertEqual(ids[M['seeds']['display']].tolist(), np.intersect1d(seeds, ids).tolist())
        self.assertEqual(M['times_s'], [float(t) for t in np.arange(4)/30])
        self.assertIn(M['probability_semantics'], [b['code'] for b in M['banners']])
        L = M['layers']['gamma']; self.assertEqual((L['title'], L['lo'], L['hi']), ('UNCALIBRATED P(moving)', 0., 1.)); self.assertTrue(L['noinfo_label'].startswith('no information'))

    def test_observed_clouds_rebuilt_like_the_runner(self):
        M = self.meta(); lc = {r['frame']: r for r in json.loads((self.rd/'svgd_joint_k1'/'loader_counts.json').read_text())}
        self.assertEqual([o['k'] for o in M['observed']['frames']], list(range(M['K'])))
        for o in M['observed']['frames']:
            F = read_js(self.rd/'debug_k1'/'data'/f"k1_obs_f{o['k']:04d}.js")
            self.assertTrue(F['matches_run_cloud']); self.assertEqual((F['cloud_sha256'], F['M_t']), (lc[o['k']]['cloud_sha256'], lc[o['k']]['M']))
            self.assertEqual(len(dec(F['xyz'], np.float32)), 3*F['M_display'])
        num = read_js(self.rd/'debug_k1'/'data'/'k1_numeric.js'); gam = np.load(self.rd/'svgd_joint_k1'/'gamma_provisional.npy')
        self.assertEqual(sum(num['hist']['gamma']['informative'])+sum(num['hist']['gamma']['no_information']), len(gam))
        self.assertIsNone(num['oracle']) if not (self.rd/'posthoc').is_dir() else None

    def test_display_subset_deterministic_and_capped(self):
        r2 = self.tmp/'run_cap'; shutil.copytree(self.rd, r2, ignore=shutil.ignore_patterns('debug_k1'))
        from debug.svgd_joint_k1_debug import write_debug
        before = tree_hashes(r2); write_debug(r2, 'full', display_cap=200); a = self.meta(r2); write_debug(r2, 'full', display_cap=200); b = self.meta(r2)
        self.assertEqual(before, tree_hashes(r2))
        self.assertEqual(a['display_ids_sha256'], b['display_ids_sha256']); self.assertLessEqual(a['N_display'], 200+len(a['seeds']['display'])+len((a['map_seeds'] or {}).get('display', [])))
        f0 = [o for o in a['observed']['frames'] if o['k'] == 0][0]; self.assertLessEqual(f0['M_display'], 200); self.assertGreater(a['observed']['voxel_edge_m'], 0)
        with self.assertRaises(ValueError): write_debug(r2, 'none')

    def test_posthoc_gt_layer_only_with_oracle_folder(self):
        M = self.meta()
        if not (self.rd/'posthoc').is_dir(): self.assertFalse(M['gt']['available']); self.assertNotIn('gt', M['layers'])
        r3 = self.tmp/'run_gt'; shutil.copytree(self.rd, r3, ignore=shutil.ignore_patterns('debug_k1'))
        if not (r3/'posthoc'/'k1_oracle'/'oracle.json').is_file():   # labelled fixture folder: the GT layer itself is recomputed from the synthetic fixture
            (r3/'posthoc'/'k1_oracle').mkdir(parents=True); (r3/'posthoc'/'k1_oracle'/'oracle.json').write_text(json.dumps({'status': 'TEST_FIXTURE_NOT_AN_ORACLE'}))
        from debug.svgd_joint_k1_debug import write_debug
        write_debug(r3, 'light'); G = self.meta(r3)
        self.assertTrue(G['gt']['available'], G['gt']['status']); self.assertIn('POST-HOC', G['layers']['gt']['name']); self.assertIn('POST-HOC', [b['code'] for b in G['banners']])
        from tests.svgd_fixtures import SyntheticDataset
        from tools.audit_svgd_joint_k1_oracles import gt_for
        data = SyntheticDataset(**json.loads((r3/'synthetic_fixture.json').read_text())); xyz = np.load(r3/'svgd_joint_k1'/'reference_catalogue.npz')['xyz']
        mv, poses, _ = gt_for(r3, None, True, data, xyz); ids = dec(G['display_ids'], np.int32)
        np.testing.assert_array_equal(dec(G['gt_moving'], np.uint8).astype(bool), np.asarray(mv, bool)[ids]); np.testing.assert_allclose(G['gt']['poses'], np.asarray(poses))

    def test_browser_qa(self):
        if not Path(CHROME).exists(): self.skipTest(f'{CHROME} not found: browser QA not run (BROWSER_UNAVAILABLE)')
        p = subprocess.run([sys.executable, '-B', str(REPO/'tools/qa_svgd_joint_k1_debug.py'), '--run', str(self.rd)], cwd=REPO, env=ENV, capture_output=True, text=True, timeout=900)
        rep = json.loads((self.rd/'debug_k1'/'qa'/'qa_report.json').read_text())
        if rep['status'] in ('BROWSER_UNAVAILABLE', 'NO_WEBGL'): self.skipTest(f"{rep['status']}: {rep.get('reason')}")
        self.assertEqual(rep['status'], 'PASS', [c for c in rep['checks'] if not c['pass']] or p.stdout[-2000:])
        self.assertEqual(p.returncode, 0); self.assertGreater(len(rep['screenshots']), 5)
        for sh in rep['screenshots']: self.assertEqual(hashlib.sha256(Path(sh['path']).read_bytes()).hexdigest(), sh['sha256'])
        self.assertEqual(self.before, tree_hashes(self.rd), 'QA wrote outside RUN/debug_k1')


if __name__ == '__main__':
    unittest.main()
