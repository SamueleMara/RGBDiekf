"""Shared tqdm display; runtime presentation never enters estimator settings."""
import os
import sys

from tqdm import tqdm


def progress(iterable=None, **kwargs):
    kwargs.setdefault("disable", os.environ.get("TQDM_DISABLE", "").lower() in ("1", "true", "yes"))
    kwargs.setdefault("file", sys.stderr)
    kwargs.setdefault("dynamic_ncols", True)
    kwargs.setdefault("mininterval", .5)
    kwargs.setdefault("delay", .2)
    kwargs.setdefault("leave", False)
    return tqdm(iterable, **kwargs)


def write(message):
    tqdm.write(message, file=sys.stdout)
    sys.stdout.flush()
