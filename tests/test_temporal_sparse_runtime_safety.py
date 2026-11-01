import errno
import json
import os
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest.mock import patch
import numpy as np
from utils.temporal_output import VerifiedFrameOutput, require_valid_frame, verify_npy, check_output_capacity, atomic_json
from tools.temporal_batch_runtime import run_child, execute_batch, persist_batch_summary


class RuntimeSafetyTests(unittest.TestCase):
    def test_hard_deadline_kills_child_group(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            code = 'import subprocess,sys,time; p=subprocess.Popen([sys.executable,"-c","import time; time.sleep(60)"]); print(p.pid,flush=True); time.sleep(60)'
            t = time.monotonic()
            result = run_child([sys.executable, '-c', code], root, root/'child.log', .3, .1, echo=False)
            self.assertEqual(result['status'], 'OS_HARD_TIMEOUT')
            self.assertLess(time.monotonic()-t, 3.)
            pid = int((root/'child.log').read_text().splitlines()[0])
            stat = Path(f'/proc/{pid}/stat')
            self.assertTrue(not stat.exists() or stat.read_text().split()[2] == 'Z')

    def test_fail_fast_and_final_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); out=root/'out'; fallback=root/'fallback';out.mkdir()
            marker=root/'unexpected'
            commands=[{'argv':[sys.executable,'-c','raise SystemExit(7)'],'output':str(root/'first'),'log':'first.log','status':'NOT_RUN','returncode':None},
                      {'argv':[sys.executable,'-c',f'open({str(marker)!r},"w").write("bad")'],'output':str(root/'second'),'log':'second.log','status':'NOT_RUN','returncode':None}]
            self.assertEqual(execute_batch(commands,out,fallback,root,2.,.1),1)
            self.assertFalse(marker.exists())
            for folder in [out,fallback]:
                data=json.loads((folder/'batch_summary.json').read_text())
                self.assertEqual(data[0]['returncode'],7);self.assertEqual(data[1]['status'],'NOT_RUN')

    def test_ENOSPC_summary_has_independent_fallback(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);out=root/'disk';fallback=root/'fallback'
            def write(path,value):
                if Path(path).parent==out:raise OSError(errno.ENOSPC,'simulated disk full')
                atomic_json(path,value)
            with patch('tools.temporal_batch_runtime.atomic_json',side_effect=write):
                dest=persist_batch_summary(out,fallback,[{'status':'NOT_RUN'}])
            self.assertEqual(Path(dest),fallback/'batch_summary.json')
            self.assertIn('simulated disk full',(fallback/'output_failure.json').read_text())
            with patch('utils.temporal_output.shutil.disk_usage',return_value=type('D',(),{'free':12})()):
                with self.assertRaises(OSError) as error:check_output_capacity(root,10,3)
                self.assertEqual(error.exception.errno,errno.ENOSPC)

    def test_nan_tails_partial_validity_reverse_order_and_truncation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            specs={'posterior_xyz':((3,2,3),'float64'),'position_variance_diag':((3,2,3),'float64')}
            writer=VerifiedFrameOutput(root,specs,[2,1,0],headroom_bytes=0)
            self.assertTrue(np.isnan(writer.arrays['posterior_xyz']).all())
            for array in writer.arrays.values():array[2]=1.
            writer.publish_verified_frame(2)
            require_valid_frame(root,2)
            with self.assertRaisesRegex(ValueError,'INVALID_TEMPORAL_FRAME'):require_valid_frame(root,1)
            with self.assertRaises(ValueError):writer.publish_verified_frame(1)
            np.testing.assert_array_equal(np.load(root/'frame_valid.npy'),[False,False,True])
            self.assertEqual(json.loads((root/'completed_frames.json').read_text())['last_completed_frame'],2)
            path=root/'posterior_xyz.npy'
            with path.open('r+b') as stream:stream.truncate(path.stat().st_size-8)
            with self.assertRaisesRegex(ValueError,'TRUNCATED'):verify_npy(path)
            with self.assertRaisesRegex(ValueError,'TRUNCATED'):require_valid_frame(root,2)

    def test_failed_diagnostic_does_not_publish_posterior(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);w=VerifiedFrameOutput(root,{'posterior_xyz':((2,1,3),'float64')},[0,1],0)
            w.arrays['posterior_xyz'][0]=0.;w.publish_verified_frame(0)
            w.arrays['posterior_xyz'][1]=1.
            with self.assertRaisesRegex(ValueError,'diagnostic'):w.publish_verified_frame(1,root/'missing.json')
            self.assertEqual(json.loads((root/'completed_frames.json').read_text())['completed_frames'],[0])

    def test_historical_failed_temporal_arrays_are_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            (root/'run_manifest.json').write_text(json.dumps({'observation_model':'temporal-rgbd'}))
            (root/'status.json').write_text(json.dumps({'status':'FULL_RUNTIME_LIMIT'}))
            with self.assertRaisesRegex(ValueError,'historical output'):require_valid_frame(root,0)

    def test_large_run_gate_requires_current_three_frame_evidence(self):
        from tools.temporal_batch_runtime import SPARSE_PRODUCTION_SOURCES,sparse_source_hashes,require_three_frame_full_cloud_gate
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for name in SPARSE_PRODUCTION_SOURCES:
                p=root/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('fixture')
            with self.assertRaisesRegex(ValueError,'NOT_ACCEPTED'):require_three_frame_full_cloud_gate(root)
            report=root/'reports/temporal_sparse_gpu_validation.json';report.parent.mkdir()
            report.write_text(json.dumps({'G4_three_frame_full_cloud_passed':True,'production_source_sha256':sparse_source_hashes(root)}))
            require_three_frame_full_cloud_gate(root)
            (root/SPARSE_PRODUCTION_SOURCES[0]).write_text('changed')
            with self.assertRaisesRegex(ValueError,'NOT_ACCEPTED'):require_three_frame_full_cloud_gate(root)

    def test_viewer_filters_uncommitted_frame_and_ENOSPC_commit_is_atomic(self):
        from debug.temporal_debug import write_temporal_dashboard
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);writer=VerifiedFrameOutput(root,{'posterior_xyz':((2,1,3),'float64')},[0,1],0)
            writer.arrays['posterior_xyz'][0]=1.;writer.publish_verified_frame(0)
            writer.arrays['posterior_xyz'][1]=2.;diagnostic=root/'frame1.json';diagnostic.write_text('{}')
            with patch('utils.temporal_output.atomic_json',side_effect=OSError(errno.ENOSPC,'simulated commit failure')):
                with self.assertRaises(OSError):writer.publish_verified_frame(1,diagnostic)
            self.assertEqual(json.loads((root/'completed_frames.json').read_text())['completed_frames'],[0])
            folder=root/'temporal/debug_frames';folder.mkdir(parents=True)
            for k in [0,1]:(folder/f'frame_{k:04d}_cam0.png').touch()
            write_temporal_dashboard(root,{'temporal_mode':'full'},[{'frame':1}])
            js=(root/'temporal/dashboard.js').read_text()
            data=json.loads(js.removeprefix('window.TEMPORAL=').removesuffix(';'))
            self.assertEqual(data['debug_frames'],[0]);self.assertEqual(data['history'],[])


if __name__=='__main__':unittest.main()
