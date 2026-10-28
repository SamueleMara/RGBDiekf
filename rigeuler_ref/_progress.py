"""Stand-in for RigEuler3DGS utils.progress.pbar (terminal progress only; no numerical effect). Silent unless RGBD_PROGRESS=1."""
import os
from tqdm.auto import tqdm


def pbar(items, description, total=None):
    return tqdm(items, desc=description, total=total, disable=os.environ.get('RGBD_PROGRESS') != '1')
