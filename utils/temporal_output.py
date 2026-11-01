"""Committed temporal frames are authoritative; unwritten array slots are never estimates."""
import errno
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
import numpy as np


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix='.' + path.name, dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as stream:
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def check_output_capacity(path, required_bytes, headroom_bytes):
    path = Path(path)
    if min(required_bytes, headroom_bytes) < 0:
        raise ValueError('Negative output size or operational headroom')
    while not path.exists():
        path = path.parent
    available = shutil.disk_usage(path).free
    if required_bytes + headroom_bytes > available:
        raise OSError(errno.ENOSPC, f'OUTPUT_CAPACITY: need {required_bytes} bytes plus '
                      f'{headroom_bytes} operational headroom; available {available}', str(path))
    return {'required_bytes': required_bytes, 'operational_headroom_bytes': headroom_bytes,
            'available_bytes': available}


def verify_npy(path):
    path = Path(path)
    with path.open('rb') as stream:
        version = np.lib.format.read_magic(stream)
        shape, order, dtype = np.lib.format._read_array_header(stream, version)
        expected = stream.tell() + math.prod(shape) * dtype.itemsize
    if dtype.hasobject or path.stat().st_size != expected:
        raise ValueError(f'TRUNCATED_OR_INVALID_NPY: {path}; expected {expected}, actual {path.stat().st_size}')
    return shape, dtype


def require_valid_frame(root, frame):
    root = Path(root)
    journal = root / 'completed_frames.json'
    if not journal.exists():
        if (root / 'run_manifest.json').exists():
            manifest = json.loads((root / 'run_manifest.json').read_text())
            if manifest.get('observation_model') == 'temporal-rgbd':
                status = json.loads((root / 'status.json').read_text()) if (root / 'status.json').exists() else {}
                if status.get('status') != 'COMPLETE':
                    raise ValueError('INVALID_TEMPORAL_FRAME: historical output has no committed-frame journal')
        return
    value = json.loads(journal.read_text())
    if frame not in value['completed_frames']:
        raise ValueError(f'INVALID_TEMPORAL_FRAME: {frame} has no committed posterior')
    for name, digest in value['frame_sha256'][str(frame)].items():
        verify_npy(root / (name + '.npy'))
        array = np.load(root / (name + '.npy'), mmap_mode='r')
        actual = hashlib.sha256(np.ascontiguousarray(array[frame]).tobytes()).hexdigest()
        if actual != digest:
            raise ValueError(f'CORRUPT_TEMPORAL_FRAME: {frame}, {name}')


class VerifiedFrameOutput:
    def __init__(self, root, specs, order, headroom_bytes=128 * 1024**2, extra_required_bytes=0):
        self.root, self.order, self.specs = Path(root), list(order), specs
        self.root.mkdir(parents=True, exist_ok=True)
        required = sum(math.prod(shape) * np.dtype(dtype).itemsize + 512 for shape, dtype in specs.values())+extra_required_bytes
        self.capacity = check_output_capacity(root, required, headroom_bytes)
        self.arrays = {}
        self.journal = {'status': 'PARTIAL', 'processing_order': self.order, 'completed_frames': [],
                        'last_completed_frame': None, 'frame_sha256': {}, 'capacity': self.capacity}
        atomic_json(self.root / 'completed_frames.json', self.journal)
        for name, (shape, dtype) in specs.items():
            array = np.lib.format.open_memmap(self.root / (name + '.npy'), mode='w+', dtype=dtype, shape=shape)
            array[:] = np.nan if np.issubdtype(np.dtype(dtype), np.floating) else 0
            array.flush()
            self.arrays[name] = array
        np.save(self.root / 'frame_valid.npy', np.zeros(len(self.order), dtype=bool))

    def publish_verified_frame(self, frame, diagnostic_path=None):
        if frame != self.order[len(self.journal['completed_frames'])]:
            raise ValueError('Frame publication must follow actual processing order')
        if len(self.journal['completed_frames']) and (diagnostic_path is None or not Path(diagnostic_path).is_file()):
            raise ValueError('No posterior diagnostic record for filtered frame')
        digests = {}
        for name, array in self.arrays.items():
            values = np.asarray(array[frame])
            if name in ('posterior_xyz', 'predicted_xyz', 'velocity', 'position_variance_diag', 'velocity_variance_diag','posterior_covariance') and not np.isfinite(values).all():
                raise ValueError(f'NONFINITE_POSTERIOR_PUBLICATION: {frame}, {name}')
            if name in ('position_variance_diag', 'velocity_variance_diag') and (values < 0).any():
                raise ValueError(f'NEGATIVE_POSTERIOR_VARIANCE: {frame}, {name}')
            array.flush()
            fd=os.open(self.root/(name+'.npy'),os.O_RDONLY)
            try:os.fsync(fd)
            finally:os.close(fd)
            verify_npy(self.root / (name + '.npy'))
            digests[name] = hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()
        journal = {**self.journal, 'completed_frames': self.journal['completed_frames'] + [frame],
                   'last_completed_frame': frame,
                   'frame_sha256': {**self.journal['frame_sha256'], str(frame): digests}}
        if len(journal['completed_frames']) == len(self.order):
            journal['status'] = 'COMPLETE'
        atomic_json(self.root / 'completed_frames.json', journal)
        self.journal = journal
        # The JSON journal is authoritative if a crash separates these two atomic writes.
        valid = np.zeros(len(self.order), dtype=bool)
        valid[journal['completed_frames']] = True
        fd, temporary = tempfile.mkstemp(prefix='.frame_valid.', dir=self.root)
        try:
            with os.fdopen(fd, 'wb') as stream:
                np.save(stream, valid)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self.root / 'frame_valid.npy')
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

