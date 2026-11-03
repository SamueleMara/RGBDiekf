"""Atomic, checksummed artifact writes for the SVGD support run (spec 10.3): write to a temporary sibling, fsync, rename;
every committed file's SHA-256 enters svgd_support/artifact_ledger.json so a partial file is never mistaken for output."""
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import numpy as np


def _commit(path, data):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f'.{path.name}.partial')
    with open(tmp, 'wb') as f: f.write(data); f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)
    return hashlib.sha256(data).hexdigest()


def _clean(o):
    if isinstance(o, dict): return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [_clean(v) for v in o]
    if isinstance(o, np.ndarray): return _clean(o.tolist())
    if isinstance(o, (np.integer,)): return int(o)
    if isinstance(o, (np.floating, float)): return None if not np.isfinite(o) else float(o)
    if isinstance(o, Path): return str(o)
    return o


class Ledger:
    def __init__(self, root):
        self.root = Path(root); self.entries = {}

    def _rec(self, path, digest):
        self.entries[str(Path(path).relative_to(self.root))] = digest; return digest

    def json(self, path, obj): return self._rec(path, _commit(path, json.dumps(_clean(obj), indent=1).encode()))

    def npy(self, path, arr):
        b = io.BytesIO(); np.save(b, np.asarray(arr), allow_pickle=False); return self._rec(path, _commit(path, b.getvalue()))

    def npz(self, path, **arrays):
        b = io.BytesIO(); np.savez_compressed(b, **{k: np.asarray(v) for k, v in arrays.items()}); return self._rec(path, _commit(path, b.getvalue()))

    def csv(self, path, rows):
        if not rows: return self._rec(path, _commit(path, b''))
        keys = list(dict.fromkeys(k for r in rows for k in r)); s = io.StringIO()
        w = csv.DictWriter(s, keys); w.writeheader(); w.writerows(rows)
        return self._rec(path, _commit(path, s.getvalue().encode()))

    def text(self, path, text): return self._rec(path, _commit(path, text.encode()))

    def close(self, path): _commit(path, json.dumps(self.entries, indent=1, sort_keys=True).encode())


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''): h.update(b)
    return h.hexdigest()
