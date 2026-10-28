"""Atomic JSON output and JSON-safe conversion of result dictionaries."""
import json
from pathlib import Path
import numpy as np


def json_write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name(path.name+'.pending')
    pending.write_text(json.dumps(value, indent=2, allow_nan=False, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else str(x))+'\n')
    pending.replace(path)


def serial(value):
    if isinstance(value, np.ndarray): return serial(value.tolist())
    if isinstance(value, (list, tuple)): return [serial(x) for x in value]
    if isinstance(value, dict): return {k: serial(v) for k, v in value.items()}
    if isinstance(value, (float, np.floating)) and not np.isfinite(value): return None
    if isinstance(value, np.generic): return value.item()
    return value
