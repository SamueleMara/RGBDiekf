"""Chunked on-disk histories of the Phase-1 temporal solver (RGBDiekf_Phase1 spec P1.2.7, P1.9, section 6 memory budget).

Every array is a numpy memmap indexed [frame (original chronological index), point, ...] and written point-chunk by
point-chunk or frame by frame; nothing all-points x all-frames is ever held on the GPU. Means are float64; full 6x6
covariances are COMPUTED in float64 and STORED as float32 (300k points x 60 frames x 36 x 8 B = 5.2 GB per float64
history would not fit the available disk next to the other histories); the storage dtype of every array is recorded in
history_manifest.json. The predicted history is not stored separately: it is exactly s_{k+1|k} = F_k s_{k|k},
P_{k+1|k} = F_k P_{k|k} F_k^T + Q_k with the stored F_k, Q_k (predicted() below reconstructs it at storage precision);
storing it would duplicate a multi-gigabyte array the filtered history already determines.
"""
import json
from pathlib import Path
import numpy as np


class History:
    def __init__(self, root, K, N, C):
        self.root = Path(root); self.root.mkdir(parents=True, exist_ok=True)
        self.K, self.N, self.C = K, N, C
        self.arrays, self.meta = {}, {}

    def create(self, name, shape, dtype, doc):
        a = np.lib.format.open_memmap(self.root/f'{name}.npy', mode='w+', dtype=dtype, shape=shape)
        self.arrays[name] = a; self.meta[name] = {'shape': list(shape), 'dtype': np.dtype(dtype).name, 'doc': doc}
        return a

    def __getitem__(self, name):
        return self.arrays[name]

    def flush(self, extra=None):
        for a in self.arrays.values(): a.flush()
        (self.root/'history_manifest.json').write_text(json.dumps({'arrays': self.meta, 'indexing': '[original chronological frame, point id, ...]',
                                                                   'covariance_storage': 'computed float64, stored float32 where the dtype says so',
                                                                   **(extra or {})}, indent=1))


def predicted(filter_mean, filter_cov, F, Q, k):
    """s_{k|k-1}, P_{k|k-1} for processing step k >= 1 from the stored filtered history (exact algebra, storage precision)."""
    m = filter_mean[k-1]@F[k-1].T
    P = F[k-1]@filter_cov[k-1].astype(np.float64)@F[k-1].T+Q[k-1]
    return m, (P+np.swapaxes(P, -1, -2))/2
