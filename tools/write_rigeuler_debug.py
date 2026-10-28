#!/usr/bin/env python3
"""Write the RigEuler3DGS debug pages for a finished rendered-image RGB-D run (see debug/rigeuler_pages.py)."""
import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scene.dataset import Dataset
from debug.rigeuler_pages import write


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run', type=Path, required=True); ap.add_argument('--input', '--scene', dest='scene', type=Path, required=True)
    ap.add_argument('--fps', type=float); ap.add_argument('--camera', type=int, default=0, help='camera of rigid_renders.html')
    ap.add_argument('--device', default='cuda'); ap.add_argument('--no-gt', action='store_true', help='skip the evaluation-only GT layers')
    a = ap.parse_args()
    data = Dataset(a.scene, fps=a.fps)
    index, pages = write(a.run, data, None if a.no_gt else a.scene, a.camera, a.device)
    for anchor, ps in pages.items():
        print(anchor); [print('  ', p) for p in ps]
    print(index)


if __name__ == '__main__':
    raise SystemExit(main())
