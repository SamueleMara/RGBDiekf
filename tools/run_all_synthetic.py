#!/usr/bin/env python3
"""Run the rendered-image RGB-D pipeline over every synthetic (SAPIEN-rendered) scene under one or more roots.

A scene is any directory containing pointcloud/joint_metadata.json and rgb/. Each scene runs sequentially (the GPU is
shared) through tools/run_rgbd_image_iekf.py into <out>/<root name>/<scene>/, then tools/evaluate_image_iekf_gt.py
scores it AFTER inference (ground truth is never read by the pipeline). A failing scene is recorded and the batch
continues. Existing complete outputs are skipped, never overwritten. Summary: <out>/summary.csv and summary.json.

Example:
  python tools/run_all_synthetic.py --out /media/.../results_25_10_26/RGBDekf/all_synthetic \\
      --roots /media/.../Sapien/output/preparation_8cams/partnet_subset_fixed_render_linear
"""
import argparse, csv, json, subprocess, sys, time
from pathlib import Path
REPO = Path(__file__).resolve().parents[1]
PY = sys.executable


def scenes(root):
    return sorted(p.parent.parent for p in Path(root).glob('*/pointcloud/joint_metadata.json') if (p.parent.parent/'rgb').is_dir())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--roots', nargs='+', type=Path, required=True); ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--fps', type=float, default=59.); ap.add_argument('--anchor', default='both', choices=['first', 'last', 'both'])
    ap.add_argument('--only', nargs='*', default=None, help='scene names to include'); ap.add_argument('--skip', nargs='*', default=[])
    ap.add_argument('--device', default='cuda'); ap.add_argument('--extra', nargs=argparse.REMAINDER, default=[], help='extra runner arguments')
    a = ap.parse_args()
    rows = []
    for root in a.roots:
        for scene in scenes(root):
            if (a.only and scene.name not in a.only) or scene.name in a.skip: continue
            out = a.out/root.name/scene.name
            row = {'root': root.name, 'scene': scene.name, 'output': str(out)}
            if (out/'result.json').exists():
                row['status'] = 'SKIPPED_EXISTING'
            elif out.exists() and any(out.iterdir()):
                row['status'] = 'SKIPPED_PARTIAL_OUTPUT'                  # never overwrite: inspect or remove by hand
            else:
                out.parent.mkdir(parents=True, exist_ok=True)
                start = time.time()
                cmd = [PY, '-B', str(REPO/'tools/run_rgbd_image_iekf.py'), '--scene', str(scene), '--output', str(out), '--fps', str(a.fps),
                       '--anchor', a.anchor, '--device', a.device, *a.extra]
                print('RUN', ' '.join(cmd), flush=True)
                with open(a.out/f'{root.name}__{scene.name}.log', 'w') as log:
                    rc = subprocess.call(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=REPO)
                row.update(returncode=rc, minutes=round((time.time()-start)/60, 1))
                row['status'] = json.loads((out/'result.json').read_text()).get('status') if (out/'result.json').exists() else 'FAILED'
            if (out/'result.json').exists() and not (out/'gt_evaluation.json').exists():
                subprocess.call([PY, str(REPO/'tools/evaluate_image_iekf_gt.py'), '--run', str(out), '--scene', str(scene), '--out', str(out/'gt_evaluation.json')],
                                stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT, cwd=REPO)
            if (out/'gt_evaluation.json').exists():
                ev = json.loads((out/'gt_evaluation.json').read_text()); row['gt_type'] = ev.get('gt_type')
                for d, r in ev.get('directions', {}).items():
                    j = r.get('joint', {}); row[f'{d}_joint'] = j.get('status')
                    t = ev.get('gt_type')
                    if t in j: row[f'{d}_axis_error_deg'] = j[t].get('axis_error_deg'); row[f'{d}_q_rmse'] = j[t].get('q_rmse')
                    c = r.get('core') or {}; row[f'{d}_core_precision'] = c.get('precision_labelled'); row[f'{d}_core_recall'] = c.get('recall_labelled')
            rows.append(row); print(json.dumps(row), flush=True)
            (a.out/'summary.json').write_text(json.dumps(rows, indent=1))
            keys = sorted({k for r in rows for k in r})
            with open(a.out/'summary.csv', 'w', newline='') as f:
                w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)


if __name__ == '__main__':
    main()
