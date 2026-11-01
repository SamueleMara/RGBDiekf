"""OS deadline and independently persisted status for temporal scene children."""
import json
import hashlib
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
from utils.temporal_output import atomic_json


SPARSE_PRODUCTION_SOURCES = tuple('estimator/'+name+'.py' for name in (
    'temporal_sparse_gpu','temporal_sparse_solve','temporal_cudss','temporal_sparse_marginalization',
    'temporal_sparse_nonlinear','temporal_sparse_run','temporal_zbuffer','temporal_rgbd_terms','temporal_reference',
    'point_kalman_cuda','projective_sensor')) + ('train.py','tools/run_temporal_rgbd_critical.py',
    'tools/temporal_batch_runtime.py','utils/temporal_output.py')


def sparse_source_hashes(repo):
    return {p:hashlib.sha256((Path(repo)/p).read_bytes()).hexdigest() for p in SPARSE_PRODUCTION_SOURCES}


def require_three_frame_full_cloud_gate(repo):
    path=Path(repo)/'reports/temporal_sparse_gpu_validation.json'
    try:
        report=json.loads(path.read_text())
        if report.get('G4_three_frame_full_cloud_passed') is True and report.get('production_source_sha256')==sparse_source_hashes(repo):return
    except (OSError,ValueError):pass
    raise ValueError('FULL_SPARSE_GPU_NOT_ACCEPTED: only bounded <=3-frame pilots on at most two scenes are enabled until USB and Blade full-cloud eight-camera filtering and real-component oracle parity pass for these sources. See '+str(path))


def terminate_group(process, grace_seconds):
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=grace_seconds)
    except subprocess.TimeoutExpired:
        pass
    # Children may outlive a parent that handled SIGTERM; terminate the entire owned group.
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait()


def run_child(command, cwd, log, timeout_seconds=None, grace_seconds=2., env=None, echo=True):
    started = time.monotonic()
    problem = []
    with Path(log).open('w') as stream:
        process = subprocess.Popen(command, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, bufsize=1, env=env, start_new_session=True)
        def copy_log():
            try:
                for line in process.stdout:
                    stream.write(line)
                    stream.flush()
                    if echo:
                        sys.stdout.write(line)
                        sys.stdout.flush()
            except OSError as error:
                problem.append(str(error))
        reader = threading.Thread(target=copy_log, daemon=True)
        reader.start()
        reason = None
        try:
            while process.poll() is None:
                if problem:
                    reason = 'OUTPUT_IO_FAILURE'
                    break
                remaining = None if timeout_seconds is None else timeout_seconds - (time.monotonic() - started)
                if remaining is not None and remaining <= 0:
                    reason = 'OS_HARD_TIMEOUT'
                    break
                try:
                    process.wait(timeout=min(.1, remaining) if remaining is not None else .1)
                except subprocess.TimeoutExpired:
                    continue
        finally:
            if process.poll() is None or reason is not None:
                terminate_group(process, grace_seconds)
            else:
                # Also reap escaped-in-lifetime group members after an ordinary parent exit.
                terminate_group(process, grace_seconds)
            reader.join(timeout=grace_seconds + 1.)
            process.stdout.close()
        if problem and reason is None:
            reason = 'OUTPUT_IO_FAILURE'
        return {'returncode': process.returncode, 'status': reason or ('CHILD_EXITED' if process.returncode == 0 else 'CHILD_FAILED'),
                'signal': -process.returncode if process.returncode < 0 else None,
                'elapsed_wall_seconds': time.monotonic() - started, 'log_error': problem or None,
                'timeout_seconds': timeout_seconds, 'process_group': process.pid}


def persist_batch_summary(output, fallback, records):
    # Always leave a recoverable journal on the independent status filesystem.
    atomic_json(Path(fallback) / 'batch_summary.json', records)
    try:
        atomic_json(Path(output) / 'batch_summary.json', records)
    except OSError as error:
        atomic_json(Path(fallback) / 'output_failure.json', {'output': str(output), 'error': str(error)})
        return str(Path(fallback) / 'batch_summary.json')
    return str(Path(output) / 'batch_summary.json')


def execute_batch(records, output, fallback, cwd, timeout_seconds=None, grace_seconds=2., continue_on_error=False):
    summary = None
    try:
        for record in records:
            record['status'] = 'RUNNING'
            persist_batch_summary(output, fallback, records)
            try:
                result = run_child(record['argv'], cwd, Path(output) / record['log'], timeout_seconds,
                                   grace_seconds, env={**os.environ, 'PYTHONUNBUFFERED': '1'})
                record.update(result)
                manifest = Path(record['output']) / 'result.json'
                if result['returncode'] == 0 and manifest.exists():
                    outcome = json.loads(manifest.read_text())
                    if outcome.get('status') != 'COMPLETE':
                        record.update(status='INCOMPLETE_FILTERING', returncode=3)
                failure = Path(record['output']) / 'failure.json'
                if failure.exists():
                    record['failure'] = json.loads(failure.read_text())
                record['completed_frame_journals'] = {}
                for direction in ('forward', 'backward'):
                    journal = Path(record['output']) / f'free_{direction}' / 'completed_frames.json'
                    if journal.exists():
                        record['completed_frame_journals'][direction] = json.loads(journal.read_text())
            except (OSError, ValueError) as error:
                record.update(status='OUTPUT_OR_CHILD_FAILURE', returncode=1, reason=str(error))
            try:
                with (Path(output)/'commands.jsonl').open('a') as stream:stream.write(json.dumps(record)+'\n')
            except OSError as error:
                record.update(status='OUTPUT_IO_FAILURE',returncode=1,reason=str(error))
            persist_batch_summary(output, fallback, records)
            failed = record.get('returncode') != 0 or record['status'] not in ('CHILD_EXITED',)
            if failed and not continue_on_error:
                break
    finally:
        for record in records:
            if record['status'] == 'RUNNING':
                record['status'] = 'INTERRUPTED'
        summary = persist_batch_summary(output, fallback, records)
        print(f'Authoritative batch summary: {summary}', flush=True)
    return 1 if any(record.get('returncode') != 0 for record in records) else 0
