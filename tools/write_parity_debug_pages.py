#!/usr/bin/env python3
"""RigEuler3DGS-style coherence debug pages for the PINNED coherence pipeline replayed on saved forward trajectories.

For each run: pinned build_supernodes(mu0.float(), 192, seed 0) -> pinned run_motion_coherence (k = 12) -> pinned
extract_moving_component (folds 5), all loaded from the 481b2cf2 Git blobs; pages written by the verbatim pinned
debug writers (rigeuler_ref/motion_coherence_debug.py, sha256 facc4f58...). No ground truth is read.
Usage: python tools/write_parity_debug_pages.py --runs v1=<run> v2=<run> --out <dir>
"""
import argparse, json, sys, tempfile
from pathlib import Path
REPO = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(REPO))
import numpy as np, torch
from tools.audit_coherence_parity import load_pinned, load_run
from rigeuler_ref import motion_coherence_debug as mcd
from debug.rigeuler_pages import _local_three


def main():
    ap = argparse.ArgumentParser(description=__doc__); ap.add_argument('--runs', nargs='+', required=True)
    ap.add_argument('--out', type=Path, required=True); ap.add_argument('--reference-repo', type=Path, default=REPO.parent/'RigEuler3DGS')
    a = ap.parse_args(); torch.set_default_dtype(torch.float64)
    mods, _ = load_pinned(a.reference_repo, tempfile.mkdtemp())
    mcs, ce, sn = mods['motion_coherence_support'], mods['coherence_extraction'], mods['supernodes']
    links = []
    for spec in a.runs:
        tag, run = spec.split('=', 1)
        v = load_run(Path(run)); X = torch.as_tensor(v['xyz']); K = X.shape[0]
        sid = sn.build_supernodes(torch.as_tensor(v['canonical']).float(), num_supernodes=192, seed=0).supernode_id.long()
        positions = [X[k] for k in range(K)]
        res = mcs.run_motion_coherence(positions, sid, knn=12, verbose=False)
        traj = mcs.supernode_trajectories(positions, sid)['trajectories'][:, res['node_ids'], :]
        ex = ce.extract_moving_component(traj, res['edge_i'], res['edge_j'], res['C'], res['num_supernodes'], folds=5, verbose=False)
        labels = ex['labels']
        res['node_class'] = labels; res['node_class_counts'] = ex['counts']
        res['labels'] = torch.as_tensor([1 if l == 'MOVING_RIGID' else 0 for l in labels])
        res['identification'] = {'status': ex['status'], 'dynamic_component': 1}; res['status'] = ex['status']
        coh = json.loads(json.dumps({k: v for k, v in ex.items() if k not in ('moving_nodes',)}, default=lambda o: o.tolist() if hasattr(o, 'tolist') else str(o)))
        for c in coh['candidates']:
            for key in ('death_merge_edge_cost', 'persistence'):
                if c.get(key) is None: c[key] = float('inf')
        res['extraction'] = coh
        d = a.out/tag; d.mkdir(parents=True, exist_ok=True)
        title = f'{tag}: pinned RigEuler coherence on saved forward trajectories'
        pages = [mcd.write_finite_strain_heatmap(d/'finite_strain_heatmap.html', res),
                 mcd.write_differential_strain_heatmap(d/'differential_strain_heatmap.html', res),
                 mcd.write_representative_edges(d/'representative_edges.html', res),
                 mcd.write_candidate_tree(d/'coherence_component_tree.html', coh, f'{title} - candidate tree'),
                 mcd.write_support_3d(d/'motion_coherence_support_3d.html', res, node_xyz=traj[0], traj=traj, gt_node=None,
                                      frame_ids=list(range(K)), title=f'{title} - support')]
        for p in pages: _local_three(Path(p))
        seed, final = ex.get('seed_core_nodes', []), ex['moving_nodes'].tolist()
        links.append((tag, run, ex['status'], len(seed), len(final), [Path(p).relative_to(a.out) for p in pages]))
        print(tag, ex['status'], 'seed', len(seed), 'final', len(final), flush=True)
    body = ''.join(f'<h2>{t}: {s}, seed {ns} nodes, final {nf}/192</h2><p>{r}</p><ul>' +
                   ''.join(f'<li><a href="{p}">{p}</a></li>' for p in ps)+'</ul>' for t, r, s, ns, nf, ps in links)
    (a.out/'index.html').write_text('<!doctype html><html><head><meta charset="utf-8"><title>Coherence parity debug</title><style>body{background:#0e1116;'
                                    'color:#e6edf3;font:14px/1.6 ui-monospace,monospace;padding:20px}a{color:#58a6ff}h2{color:#58a6ff;font-size:14px}</style>'
                                    '</head><body><h1>Pinned RigEuler3DGS 481b2cf2 coherence replayed on saved trajectories</h1>'
                                    '<p>Identical pinned partition, graph, strain and extractor; only the input trajectories differ.</p>'+body+'</body></html>')
    print(a.out/'index.html')


if __name__ == '__main__':
    main()
