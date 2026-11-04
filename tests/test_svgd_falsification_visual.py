"""Post-hoc falsification viewer (debug/svgd_falsification_debug.py): display-only and numerically faithful to the audit files.

A CPU synthetic run (tools/svgd_synthetic_run.py, translate, 6 frames, 3 cameras) and its post-hoc audit (tools/audit_svgd_falsification.py
--synthetic --skip-sensitivity: the viewer does not show the diagnostic feature models) are built once in a temporary directory; the viewer is written with two display caps. The decoded base64 arrays must equal
oracle_trajectories.npz at the displayed IDs, the numeric payload must equal the CSV / JSON files, one display subset must serve every frame,
and nothing outside A/viewer may change (every other file under RUN is hashed before / after). The headless-Chrome QA runs when Chrome
exists (else skipTest with the reason).

    python -B -m unittest tests.test_svgd_falsification_visual -v
Optional: SVGD_FALSIFICATION_TEST_DIR=<parent for the temporary run> (default: system temp), SVGD_FALSIFICATION_KEEP=1 keeps it,
SVGD_FALSIFICATION_AUDIT=<existing audit folder> reuses an audit (its RUN is still hashed; nothing is rebuilt).
"""
import hashlib, json, os, shutil, subprocess, sys, tempfile, unittest
from pathlib import Path
import numpy as np
REPO = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(REPO))
from debug.svgd_falsification_debug import write_falsification_debug, cosine_rows
from debug.svgd_support_debug import read_csv
from utils.io import serial
from tools.qa_svgd_support_debug import read_js, dec

CHROME = '/usr/bin/google-chrome'
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()


def tree(root, skip):
    return {str(p.relative_to(root)): sha(p) for p in sorted(Path(root).rglob('*')) if p.is_file() and skip not in p.parents}


class FalsificationVisual(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        env = {**os.environ, 'CUDA_VISIBLE_DEVICES': ''}; cls.tmp = None
        if os.environ.get('SVGD_FALSIFICATION_AUDIT'):
            cls.A = Path(os.environ['SVGD_FALSIFICATION_AUDIT']).resolve(); cls.rd = cls.A.parent.parent
        else:
            cls.tmp = Path(tempfile.mkdtemp(prefix='svgd_fals_', dir=os.environ.get('SVGD_FALSIFICATION_TEST_DIR'))); cls.rd = cls.tmp/'run_t'
            for cmd in ([sys.executable, '-B', str(REPO/'tools/svgd_synthetic_run.py'), '--motion', 'translate', '--frames', '6', '--n-cams', '3', '--device', 'cpu', '--output', str(cls.rd)],
                        [sys.executable, '-B', str(REPO/'tools/audit_svgd_falsification.py'), '--run', str(cls.rd), '--synthetic', '--device', 'cpu', '--skip-sensitivity']):
                p = subprocess.run(cmd, cwd=REPO, env=env, capture_output=True, text=True, timeout=1800)
                if p.returncode: raise RuntimeError(f'{cmd[2]} exit {p.returncode}: {p.stdout[-1500:]} {p.stderr[-1500:]}')
            cls.A = cls.rd/'posthoc'/'svgd_falsification'
        cls.V = cls.A/'viewer'
        if cls.V.exists(): shutil.rmtree(cls.V)
        cls.before = tree(cls.rd, cls.V)
        cls.small = write_falsification_debug(cls.A, display_cap=97); cls.M_small = read_js(cls.V/'data/falsification_meta.js')
        cls.out = write_falsification_debug(cls.A, display_cap=20000)
        cls.after = tree(cls.rd, cls.V)

    @classmethod
    def tearDownClass(cls):
        if cls.tmp and not os.environ.get('SVGD_FALSIFICATION_KEEP'): shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_files_written_only_in_viewer(self):
        self.assertEqual(self.before, self.after)                              # audit files, RUN/svgd_support, manifest: byte-identical
        for f in ('index.html', 'falsification_3d.html', 'falsification_numerical.html', 'three.min.js', 'THREE_LICENSE.txt', 'projective_common.js', 'projective_theme.css',
                  'data/falsification_meta.js', 'data/falsification_numeric.js'):
            self.assertTrue((self.V/f).is_file(), f)
        K = len(np.load(self.A/'oracle_trajectories.npz')['X_gt_POSTHOC'])
        self.assertEqual(sorted(p.name for p in (self.V/'data').glob('falsification_f*.js')), [f'falsification_f{k:04d}.js' for k in range(K)])

    def test_arrays_equal_npz_at_display_ids(self):
        Z = np.load(self.A/'oracle_trajectories.npz'); X0, Xg, mv, Phi = Z['X0'], Z['X_gt_POSTHOC'], Z['moving_POSTHOC'].astype(bool), Z['Phi']
        M = read_js(self.V/'data/falsification_meta.js'); ids = dec(M['display_ids'], np.int32).astype(int); vec = np.asarray(M['vector_ids'])
        self.assertEqual((M['N'], M['K'], M['n_moving_full']), (len(X0), len(Xg), int(mv.sum())))
        self.assertTrue(set(vec.tolist()) <= set(ids.tolist())); np.testing.assert_array_equal(ids[dec(M['vector_index'], np.int32)], vec)
        self.assertTrue(mv[vec].any() and (~mv[vec]).any())
        np.testing.assert_array_equal(dec(M['X0'], np.float32).reshape(-1, 3), X0[ids].astype(np.float32))
        np.testing.assert_array_equal(dec(M['moving'], np.uint8).astype(bool), mv[ids])
        man = json.loads((self.rd/'run_manifest.json').read_text()); self.assertEqual(M['times_s'], man['dataset']['timestamps_seconds'])
        arms = [a for a in 'ABCD' if f'X_arm_{a}' in Z.files]; self.assertEqual([r['arm'] for r in M['arms']], arms)
        for k in range(M['K']):
            F = read_js(self.V/f'data/falsification_f{k:04d}.js'); self.assertTrue(F['complete']); self.assertEqual(F['N_display'], len(ids))
            np.testing.assert_array_equal(dec(F['gt'], np.float32).reshape(-1, 3), Xg[k, ids].astype(np.float32))
            np.testing.assert_array_equal(dec(F['phi_vec'], np.float32).reshape(-1, 3), Phi[vec, k].astype(np.float32))
            np.testing.assert_array_equal(dec(F['phi_norm'], np.float32), np.linalg.norm(Phi[ids, k], axis=1).astype(np.float32))
            np.testing.assert_array_equal(dec(F['cos_gt'], np.float32), cosine_rows(Phi[ids, k], Xg[k, ids]-X0[ids]).astype(np.float32))
            for a in arms: np.testing.assert_array_equal(dec(F['arms'][a], np.float32).reshape(-1, 3), Z[f'X_arm_{a}'][k, ids].astype(np.float32))
            self.assertEqual(F['time_s'], M['times_s'][k])
        self.assertTrue(np.isnan(dec(read_js(self.V/'data/falsification_f0000.js')['cos_gt'], np.float32)).all())   # X_GT,0 == X0: undefined, never 0

    def test_display_subset_fixed_and_cap_bounded(self):
        M, S = read_js(self.V/'data/falsification_meta.js'), self.M_small
        self.assertNotEqual(M['display_ids_sha256'], S['display_ids_sha256']); self.assertLessEqual(S['N_display'], 97+len(S['vector_ids']))
        self.assertEqual(M['N_display'], M['N'])                               # cap 20000 > N: every particle shown

    def test_numeric_payload_equals_audit_files(self):
        Nm = read_js(self.V/'data/falsification_numeric.js'); S = json.loads((self.A/'falsification_summary.json').read_text())
        js = lambda rows: json.loads(json.dumps(serial(rows), allow_nan=False, default=str))      # NaN -> null exactly as write_js
        self.assertEqual(Nm['energy_oracle'], js(read_csv(self.A/'energy_oracle.csv'))); self.assertEqual(Nm['line_search'], js(read_csv(self.A/'line_search_counterfactual.csv')))
        F3 = json.loads((self.A/'stein_field_audit.json').read_text()); self.assertEqual(Nm['stein_per_frame'], js(F3['per_frame'])); self.assertEqual(Nm['fd_closure'], js(F3['fd_closure']))
        self.assertEqual(Nm['summary']['classification'], S['classification']); self.assertEqual(Nm['diagnosis_md'], (self.A/'diagnosis.md').read_text())
        eo = Nm['energy_oracle']; row = lambda c, k: next(r for r in eo if r['candidate'] == c and r['frame'] == k)
        d = [row('GT', k)['E_obs_t']-row('S', k)['E_obs_t'] for k in range(len(Nm['times_s']))]
        np.testing.assert_allclose(d, S['gate_F1_F2']['dE_obs_GT_minus_S_t'], rtol=1e-9, atol=1e-9)

    def test_bad_input_raises(self):
        with self.assertRaises(FileNotFoundError): write_falsification_debug(self.A/'does_not_exist')
        with self.assertRaises(ValueError): write_falsification_debug(self.A, display_cap=0)

    def test_headless_qa(self):
        if not Path(CHROME).exists(): self.skipTest(f'{CHROME} not found: browser QA not run (BROWSER_UNAVAILABLE)')
        env = {**os.environ, 'TMPDIR': os.environ.get('SVGD_QA_TMPDIR', '/tmp')}
        p = subprocess.run([sys.executable, '-B', str(REPO/'tools/qa_svgd_falsification_debug.py'), '--audit', str(self.A)], capture_output=True, text=True, timeout=900, env=env)
        rep = json.loads((self.V/'qa/qa_report.json').read_text())
        if rep['status'] in ('BROWSER_UNAVAILABLE', 'NO_WEBGL'): self.skipTest(f"{rep['status']}: {rep.get('reason')}")
        self.assertEqual(rep['status'], 'PASS', [c for c in rep['checks'] if not c['pass']] or p.stdout[-2000:])
        self.assertEqual(p.returncode, 0); self.assertGreater(rep['n_checks'], 30)
        for s_ in rep['screenshots']: self.assertEqual(sha(s_['path']), s_['sha256'])
        self.assertEqual(tree(self.rd, self.V), self.before)                  # QA writes only A/viewer/qa


if __name__ == '__main__':
    unittest.main()
