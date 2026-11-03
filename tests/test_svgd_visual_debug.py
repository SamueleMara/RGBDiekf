"""SVGD Stage-1 visual debug (spec 11.3): the debug payload is display-only and numerically faithful.

A tiny synthetic run (2 cameras, 4 frames, translating plate) is written twice with --debug full and different
display caps; the decoded base64 arrays of RUN/debug/data/*.js must equal the saved arrays at the displayed IDs, the
observed clouds must sit on one fixed pixel lattice, and every inference array (.npy/.npz in RUN/svgd_support) must be
byte-identical between the two display caps. Fixture copies add p_dynamic / seeds (with NaN) and remove the transport
(stopped run) to exercise the probability legend, the unresolved colour and NO_ESTIMATION. The headless-Chrome QA runs
when Chrome exists (else skipTest with the reason).

    python -B -m unittest tests.test_svgd_visual_debug -v
Optional: SVGD_DEBUG_TEST_DIR=<parent for the temporary runs>, SVGD_DEBUG_KEEP=1 keeps them.
"""
import hashlib, json, os, shutil, subprocess, sys, tempfile, unittest
from pathlib import Path
import numpy as np
REPO = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(REPO))
from tests.svgd_fixtures import SyntheticDataset, runner_args, small_config
from debug.svgd_support_debug import write_debug, lattice_keep
from tools.qa_svgd_support_debug import read_js, dec

CHROME = '/usr/bin/google-chrome'
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
inference = lambda run: {str(p.relative_to(run)): sha(p) for p in sorted((run/'svgd_support').rglob('*')) if p.suffix in ('.npy', '.npz')}
everything = lambda run: {str(p.relative_to(run)): sha(p) for p in sorted((run/'svgd_support').rglob('*')) if p.is_file()}


class SvgdVisualDebug(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from estimator.svgd_support_run import run
        cls.tmp = Path(tempfile.mkdtemp(prefix='svgd_dbg_', dir=os.environ.get('SVGD_DEBUG_TEST_DIR')))
        cls.data = SyntheticDataset(K=4, motion='translate', speed=.15, n_cams=2)
        cfg = small_config(cls.tmp/'cfg.json', REPO/'configs/svgd_support_experimental.json', **{'features.D': 256})
        cls.A, cls.B = cls.tmp/'runA', cls.tmp/'runB'
        cls.codes = [run(runner_args(str(r), cfg, debug='full', display_cap=c), data=cls.data) for r, c in ((cls.A, 400), (cls.B, 97))]

    @classmethod
    def tearDownClass(cls):
        if not os.environ.get('SVGD_DEBUG_KEEP'): shutil.rmtree(cls.tmp, ignore_errors=True)

    def copy(self, name):
        dst = self.tmp/name; shutil.rmtree(dst, ignore_errors=True); shutil.copytree(self.A, dst); shutil.rmtree(dst/'debug'); return dst

    def test_pages_written_by_runner(self):
        for r in (self.A, self.B):
            st = json.loads((r/'svgd_support/status.json').read_text())
            self.assertNotIn('debug', st, st.get('debug'))
            for f in ('index.html', 'svgd_support_3d.html', 'svgd_support_numerical.html', 'three.min.js', 'THREE_LICENSE.txt', 'projective_common.js', 'projective_theme.css',
                      *[f'data/svgd_support_{k}.js' for k in ('meta', 'numeric', 'traj', 'traces', 'f0000', 'f0001', 'f0002', 'f0003')]):
                self.assertTrue((r/'debug'/f).is_file(), f)

    def test_arrays_equal_saved_at_display_ids(self):
        sv, dd = self.A/'svgd_support', self.A/'debug/data'; M = read_js(dd/'svgd_support_meta.js'); ids = dec(M['display_ids'], np.int32).astype(int)
        can = np.load(sv/'canonical.npz'); X = np.load(sv/'transport_provisional.npy'); V = np.load(sv/'velocity_provisional.npy'); TS = np.load(sv/'transport_status.npy')
        traced = sorted(int(p.stem) for p in (sv/'trace_points').glob('*.json'))
        self.assertEqual(M['traced_ids'], traced); self.assertTrue(set(traced) <= set(ids.tolist()))
        self.assertEqual(len(np.setdiff1d(ids, traced)), len(np.setdiff1d(np.unique(np.linspace(0, M['N']-1, 400).round().astype(int)), traced)))
        np.testing.assert_array_equal(dec(M['canonical_xyz'], np.float32).reshape(-1, 3), can['xyz0'][ids].astype(np.float32))
        np.testing.assert_array_equal(dec(M['source'], np.int32).reshape(-1, 3), can['source'][ids])
        np.testing.assert_array_equal(dec(M['motion_score'], np.float64), np.load(sv/'motion_score_uncalibrated.npy')[ids])
        np.testing.assert_array_equal(dec(M['probability_status'], np.uint8), np.load(sv/'probability_status.npy')[ids])
        self.assertEqual(M['layers']['score']['name'], 'motion evidence score (UNCALIBRATED)'); self.assertNotIn('prob', M['layers'])
        self.assertIn('PROBABILITY_UNCALIBRATED', [b['code'] for b in M['banners']])
        for k in range(M['K']):
            F = read_js(dd/f'svgd_support_f{k:04d}.js'); self.assertTrue(F['complete']); self.assertEqual(F['N_display'], len(ids))
            np.testing.assert_array_equal(dec(F['transported'], np.float32).reshape(-1, 3), X[k][ids].astype(np.float32))
            np.testing.assert_array_equal(dec(F['velocity'], np.float32).reshape(-1, 3), V[k][ids].astype(np.float32))
            np.testing.assert_array_equal(dec(F['transport_status'], np.uint8), TS[k][ids])
            self.assertAlmostEqual(F['time_s'], float(self.data.times[k]), places=12)
        TJ = read_js(dd/'svgd_support_traj.js'); tx = dec(TJ['xyz'], np.float32).reshape(len(TJ['ids']), M['K'], 3)
        np.testing.assert_array_equal(tx, X[:, TJ['ids']].transpose(1, 0, 2).astype(np.float32))

    def test_observed_cloud_fixed_pixel_lattice(self):
        from scene.config import load_config
        from scene.pointcloud_sequence import build_masked_cloud
        M = read_js(self.A/'debug/data/svgd_support_meta.js'); s = M['observed']['lattice_step']; cams = M['cameras']
        self.assertGreater(s, 1)                                               # cap 400 < M_0 forces thinning
        for k in range(M['K']):
            O = read_js(self.A/f'debug/data/svgd_support_f{k:04d}.js')['observed']; cl = build_masked_cloud(self.data, k, load_config())
            pix, cam = dec(O['pixel'], np.int32), dec(O['camera'], np.uint8); w = np.asarray([c['width'] for c in cams])[cam]
            self.assertTrue(np.all(pix % w % s == 0) and np.all(pix // w % s == 0))
            keep = lattice_keep(cl, cams, s)
            self.assertEqual((O['M_t'], O['M_display']), (cl.M, int(keep.sum()))); self.assertTrue(O['matches_run_cloud'])
            np.testing.assert_array_equal(dec(O['xyz'], np.float32).reshape(-1, 3), cl.xyz[keep].astype(np.float32))
            if k == 0: self.assertLessEqual(O['M_display'], 400)

    def test_inference_independent_of_display_cap(self):
        a, b = inference(self.A), inference(self.B)
        self.assertEqual(self.codes[0], self.codes[1]); self.assertGreaterEqual(len(a), 6); self.assertEqual(a, b)
        ma, mb = (read_js(r/'debug/data/svgd_support_meta.js') for r in (self.A, self.B))
        self.assertNotEqual(ma['N_display'], mb['N_display']); self.assertNotEqual(ma['display_ids_sha256'], mb['display_ids_sha256'])

    def test_write_debug_never_touches_inference_light_keyframes(self):
        C = self.copy('runC'); before = everything(C)
        out = write_debug(C, 'light', self.data, display_cap=50)
        self.assertEqual(everything(C), before)
        self.assertEqual(out['frames'], [0, 1, 3]); self.assertEqual(sorted(p.name for p in (C/'debug/data').glob('svgd_support_f*.js')), [f'svgd_support_f{k:04d}.js' for k in (0, 1, 3)])

    def test_bad_input_raises(self):
        with self.assertRaises(ValueError): write_debug(self.A, 'none', self.data, 400)
        with self.assertRaises(FileNotFoundError): write_debug(self.tmp/'does_not_exist', 'full', self.data, 400)
        with self.assertRaises(ValueError): write_debug(self.A, 'full', self.data, -1)

    def fixture_probability(self):
        P = self.tmp/'runP'
        if (P/'debug/data/svgd_support_meta.js').is_file(): return P
        P = self.copy('runP'); sv = P/'svgd_support'; sc = np.load(sv/'motion_score_uncalibrated.npy'); N = len(sc)
        p = sc/sc.max(); p[::7] = np.nan; st = np.where(np.isfinite(p), 0, 5).astype(np.int8); st[3::11] = 3   # NaN and NON_IDENTIFIABLE -> unresolved
        np.save(sv/'p_dynamic.npy', p); np.save(sv/'p_static.npy', 1-p); np.save(sv/'probability_status.npy', st)
        np.save(sv/'reliable_moving_seed_ids.npy', np.argsort(-sc)[:5].astype(np.int64))                    # test fixture inputs, never produced by the runner
        write_debug(P, 'full', self.data, display_cap=400); return P

    def test_probability_layer_seeds_and_unresolved(self):
        P = self.fixture_probability(); sv = P/'svgd_support'; M = read_js(P/'debug/data/svgd_support_meta.js'); ids = dec(M['display_ids'], np.int32).astype(int)
        seeds = np.load(sv/'reliable_moving_seed_ids.npy'); X = np.load(sv/'transport_provisional.npy')
        self.assertTrue(set(seeds.tolist()) <= set(ids.tolist())); self.assertEqual(M['seeds']['ids'], sorted(seeds.tolist()))
        np.testing.assert_array_equal(dec(M['p_dynamic'], np.float64), np.load(sv/'p_dynamic.npy')[ids])        # NaN preserved bit-exactly
        L = M['layers']['prob']; self.assertEqual((L['lo'], L['hi'], L['lo_label'], L['hi_label']), (0., 1., '0 (static explanation)', '1 (moving explanation)'))
        self.assertIn('unresolved', L['nan_label']); self.assertNotIn(M['nan_colour'].lower(), ('#440154', '#fde725'))
        TJ = read_js(P/'debug/data/svgd_support_traj.js'); tx = dec(TJ['xyz'], np.float32).reshape(len(TJ['ids']), M['K'], 3)
        for s_ in seeds: np.testing.assert_array_equal(tx[TJ['ids'].index(int(s_))], X[:, s_].astype(np.float32))

    def fixture_stopped(self):
        Q = self.tmp/'runQ'
        if (Q/'debug/data/svgd_support_meta.js').is_file(): return Q
        Q = self.copy('runQ')
        for f in ('transport_provisional.npy', 'velocity_provisional.npy', 'transport_status.npy'): (Q/'svgd_support'/f).unlink()
        write_debug(Q, 'full', self.data, display_cap=400); return Q

    def test_stopped_run_degrades(self):
        Q = self.fixture_stopped(); M = read_js(Q/'debug/data/svgd_support_meta.js')
        self.assertEqual(M['K_transport'], 0); self.assertIn('NO_ESTIMATION', [b['code'] for b in M['banners']])
        self.assertIsNone(read_js(Q/'debug/data/svgd_support_f0001.js')['transported'])

    def test_headless_qa(self):
        if not Path(CHROME).exists(): self.skipTest(f'{CHROME} not found: browser QA not run (BROWSER_UNAVAILABLE)')
        for r in (self.A, self.fixture_probability(), self.fixture_stopped()):
            p = subprocess.run([sys.executable, '-B', str(REPO/'tools/qa_svgd_support_debug.py'), '--run', str(r)], capture_output=True, text=True, timeout=900)
            rep = json.loads((r/'debug/qa/qa_report.json').read_text())
            if rep['status'] in ('BROWSER_UNAVAILABLE', 'NO_WEBGL'): self.skipTest(f"{rep['status']}: {rep.get('reason')}")
            self.assertEqual(rep['status'], 'PASS', [c for c in rep['checks'] if not c['pass']] or p.stdout[-2000:])
            self.assertEqual(p.returncode, 0)
            for s_ in rep['screenshots']: self.assertEqual(sha(s_['path']), s_['sha256'])


if __name__ == '__main__':
    unittest.main()
