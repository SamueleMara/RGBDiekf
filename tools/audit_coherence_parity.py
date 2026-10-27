#!/usr/bin/env python3
"""Step 1 (CLAUDE_STEP1_RIGEULER_COHERENCE_SUPERNODE_PARITY_AUDIT.md): read-only coherence parity audit.

Replays saved FORWARD Phase-1 trajectories through the PINNED RigEuler3DGS 481b2cf2 coherence code (loaded from the
Git blobs, not from the vendored copies) and through the local RGB-D adaptation, holding inputs identical, and
instruments every selection boundary. CPU, float64. No Phase-1 rerun, no GT in any inference (GT is not read at all).

Comparisons (each with refine=False and refine=True):
  B1  identical partition = the v2 run's own 192-node coherence membership
  B2  reference partition = pinned build_supernodes(mu0.float(), num_supernodes=192, seed=0) on the canonical XYZ
  per partition, a 2x2 matrix: extractor {pinned, local supported_coherence} x trajectory policy
      {raw: pinned uniform supernode_trajectories on the raw filtered XYZ,
       supported: local image-supported node displacements + node-frame weights (the v2 production policy)}
      graph: pinned canonical_knn_graph(k=12) on the node trajectory at chronological time 0 (identical in all cells)
      strain: pinned extractor -> pinned edge_strain on the given trajectory; local extractor -> weighted_edge_strain
  REPLAY  the earlier (v1) forward trajectories through the identical pinned construction (B2 partition, raw, pinned)

Usage:
  python tools/audit_coherence_parity.py --v2 <run> [--v1 <run>] --reference-repo ../RigEuler3DGS --out <dir>
"""
import argparse, csv, hashlib, importlib, json, math, subprocess, sys, time
from pathlib import Path
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
COMMIT = '481b2cf2d5602932286948cf14114f9caee4de3c'
PINNED = ['estimator/supernodes.py', 'estimator/motion_coherence_support.py', 'estimator/coherence_extraction.py']
LOCAL = ['pointwise_rgbd/image_coherence.py', 'pointwise_rgbd/supported_coherence.py', 'pointwise_rgbd/rigeuler_ref/coherence_extraction.py',
         'pointwise_rgbd/supernodes.py', 'pointwise_rgbd/image_phase1.py', 'pointwise_rgbd/image_iekf.py', 'pointwise_rgbd/rgbd_point_renderer.py',
         'pointwise_rgbd/phase1_covariance.py', 'pointwise_rgbd/restricted_posterior.py', 'tools/run_rgbd_image_iekf.py']


def sha(path, chunk=1 << 24):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while (b := f.read(chunk)): h.update(b)
    return h.hexdigest()


def load_pinned(ref_repo, tmp):
    """Write the pinned blobs into tmp/estimator, with stubs for the two non-numerical imports, and import them."""
    tmp = Path(tmp); (tmp/'estimator').mkdir(parents=True, exist_ok=True); (tmp/'utils').mkdir(exist_ok=True)
    (tmp/'diff_gaussian_rasterization').mkdir(exist_ok=True)
    (tmp/'estimator/__init__.py').write_text(''); (tmp/'utils/__init__.py').write_text('')
    (tmp/'utils/progress.py').write_text('def pbar(items, *a, **k):\n    return items\n')        # terminal progress only
    (tmp/'diff_gaussian_rasterization/__init__.py').write_text('_C = None  # CUDA kernels: unused by build_supernodes\n')
    blobs = {}
    for f in PINNED:
        data = subprocess.check_output(['git', 'show', f'{COMMIT}:{f}'], cwd=ref_repo)
        blob = subprocess.check_output(['git', 'rev-parse', f'{COMMIT}:{f}'], cwd=ref_repo, text=True).strip()
        (tmp/f).write_bytes(data)
        assert hashlib.sha256((tmp/f).read_bytes()).hexdigest() == hashlib.sha256(data).hexdigest()
        blobs[f] = {'git_blob': blob, 'sha256': hashlib.sha256(data).hexdigest()}
    sys.path.insert(0, str(tmp))
    for m in [m for m in list(sys.modules) if m == 'estimator' or m.startswith('estimator.') or m == 'utils' or m.startswith('utils.')]:
        del sys.modules[m]
    mods = {name: importlib.import_module(f'estimator.{name}') for name in ('supernodes', 'motion_coherence_support', 'coherence_extraction')}
    for name, m in mods.items(): assert Path(m.__file__).resolve().parent == (tmp/'estimator').resolve(), f'{name} not loaded from the pinned blob'
    return mods, blobs


def node_pairs(ei, ej):
    a, b = np.minimum(ei, ej), np.maximum(ei, ej)
    return set(zip(a.tolist(), b.tolist()))


def refine_energies(X, omega, core, kabsch, weighted):
    """E_i^M, E_i^I of every node under the CORE-only rigid trajectory (the pinned section-19 quantities)."""
    T, n = X.shape[0], X.shape[1]
    core = torch.as_tensor(core).long()
    EM, EI = torch.zeros(n, dtype=torch.float64), torch.zeros(n, dtype=torch.float64)
    idx = torch.arange(n)
    for t in range(T):
        if weighted:
            from pointwise_rgbd.supported_coherence import _fit
            fit = _fit(X[0, core], X[t, core], omega[t, core])
            if fit is None: continue
            R, p = fit; w = omega[t]
        else:
            R, p = kabsch(X[0, core], X[t, core]); w = torch.ones(n, dtype=torch.float64)
        EM += (X[t, idx]-X[0, idx]@R.T-p).pow(2).sum(1)*w
        EI += (X[t, idx]-X[0, idx]).pow(2).sum(1)*w
    return EM, EI


def run_cell(name, X, omega, ei, ej, mods, extractor, refine, folds=5):
    """One extraction with full instrumentation."""
    mcs, ce = mods['motion_coherence_support'], mods['coherence_extraction']
    n = X.shape[1]
    if extractor == 'pinned':
        st = mcs.edge_strain(X, ei, ej); C = st['C']; e_i, e_j = ei, ej
        res = ce.extract_moving_component(X, e_i, e_j, C, n, folds=folds, refine=refine, verbose=False)
        kabsch, weighted = ce._kabsch, False
    else:
        from pointwise_rgbd import supported_coherence as sc
        st = sc.weighted_edge_strain(X, omega, ei, ej); keep = st['supported_edges']
        C, e_i, e_j = st['C'][keep], ei[keep], ej[keep]
        res = sc.extract_moving_component(X, omega, e_i, e_j, C, n, folds=folds, refine=refine)
        kabsch, weighted = ce._kabsch, True
    sel = res['selection']
    cands = []
    for c in res['candidates']:
        r, w, e = c.get('rigidity', {}), c.get('rigidity_world', {}), c.get('edges', {})
        cands.append({'candidate_id': c['candidate_id'], 'num_nodes': c['num_members'], 'persistence': c['persistence'],
                      'birth_edge_cost': c['birth_edge_cost'], 'death_merge_edge_cost': c['death_merge_edge_cost'],
                      'size_stratum_log2': int(math.log2(max(c['num_members'], 1))), 'complement_nodes': c.get('complement_nodes'),
                      'num_internal_edges': e.get('num_internal_edges'), 'num_boundary_edges': e.get('num_boundary_edges'),
                      'internal_C_q50': e.get('internal_q50'), 'boundary_C_q50': e.get('boundary_q50'),
                      'rel_eta_cv': r.get('eta_cv'), 'rel_rss_rigid_cv': r.get('rss_rigid_cv'), 'rel_rss_identity_cv': r.get('rss_identity_cv'),
                      'rel_delta_rss_cv': r.get('delta_rss_cv'), 'rel_delta_rss_cv_per_node_frame': r.get('delta_rss_cv_per_node_frame'),
                      'relative_motion_ratio': r.get('relative_motion_ratio'), 'rel_delta_bic': r.get('delta_bic'), 'rel_degenerate': bool(r.get('degenerate', False)),
                      'world_eta_cv': w.get('eta_cv'), 'world_delta_rss_cv': w.get('delta_rss_cv'),
                      'world_delta_rss_cv_per_node_frame': w.get('delta_rss_cv_per_node_frame'),
                      'qualified': c['candidate_id'] in set(sel.get('qualified_ids', [])), 'selected': c['candidate_id'] == sel.get('selected')})
    seed = list(res.get('seed_core_nodes', []))
    final = sorted(res['moving_nodes'].tolist()) if len(res['moving_nodes']) else []
    rec = {'name': name, 'extractor': extractor, 'refine': refine, 'status': res['status'], 'num_hierarchy_candidates': res['num_merge_candidates'],
           'num_pool': len(cands), 'selected': sel.get('selected'), 'num_qualified': sel.get('num_qualified'), 'selection_rule_or_reason': sel.get('rule', sel.get('reason')),
           'seed_nodes': seed, 'final_moving_nodes': final, 'num_edges_used': int(len(e_i)), 'candidates': cands, 'labels': list(res['labels'])}
    if seed:
        sc_ = next(c for c in cands if c['selected'])
        rec['seed_invariant'] = {'complement_nonempty': (sc_['complement_nodes'] or 0) > 0, 'boundary_edges_positive': (sc_['num_boundary_edges'] or 0) > 0}
        EM, EI = refine_energies(X, omega, torch.as_tensor(seed), kabsch, weighted)
        rest = np.setdiff1d(np.arange(n), seed)
        recruited = sorted(set(final)-set(seed))
        rec['refinement'] = {'E_M': EM.tolist(), 'E_I': EI.tolist(), 'recruited': recruited,
                             'rejected': sorted(set(rest.tolist())-set(recruited)),
                             'recruits_explained_by_EM_lt_EI': bool(all(EM[i] < EI[i] for i in recruited)) if refine else None,
                             'growth_fully_listed': bool(set(final) == set(seed) | set(recruited))}
    return rec


def components(n, ei, ej):
    parent = list(range(n))
    def find(a):
        while parent[a] != a: parent[a] = parent[parent[a]]; a = parent[a]
        return a
    for a, b in zip(ei.tolist(), ej.tolist()): parent[find(a)] = find(b)
    return len({find(v) for v in range(n)})


def match_partitions(sa, sb):
    """Overlap matching of two point partitions (labels are arbitrary): per group of A, best B group and its share."""
    na, nb = int(sa.max())+1, int(sb.max())+1
    M = np.zeros((na, nb), np.int64); np.add.at(M, (sa, sb), 1)
    best = M.argmax(1); share = M.max(1)/np.maximum(M.sum(1), 1)
    return {'groups_a': na, 'groups_b': nb, 'identical_up_to_relabel': bool((M > 0).sum(1).max() == 1 and (M > 0).sum(0).max() == 1),
            'purity_median': float(np.median(share)), 'purity_min': float(share.min()), 'groups_with_purity_lt_0.9': int((share < .9).sum()),
            'size_a_quantiles': np.quantile(np.bincount(sa), [0, .5, 1]).tolist(), 'size_b_quantiles': np.quantile(np.bincount(sb), [0, .5, 1]).tolist()}


def load_run(run, anchor='first'):
    d = Path(run)/anchor
    st = np.load(d/'phase1/state.npy', mmap_mode='r')
    part = np.load(d/'phase1/partition.npz')
    rj = json.loads((d/'phase1/run.json').read_text())
    if rj['order'] != sorted(rj['order']) or rj['anchor'] != 'first': raise SystemExit('forward run expected (chronological order, anchor first)')
    return {'dir': d, 'xyz': np.asarray(st[..., :3], np.float64), 'observed': np.load(d/'phase1/point_observed.npy'), 'canonical': part['xyz'],
            'phase1_membership': part['membership'], 'times': np.asarray(rj['timestamps_seconds'], np.float64),
            'reps': np.load(d/'coherence/representatives.npz', allow_pickle=True) if (d/'coherence/representatives.npz').exists() else None,
            'coherence_result': json.loads((d/'coherence/result.json').read_text()) if (d/'coherence/result.json').exists() else None}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--v2', type=Path, required=True); ap.add_argument('--v1', type=Path)
    ap.add_argument('--reference-repo', type=Path, default=REPO.parent/'RigEuler3DGS'); ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--report-dir', type=Path, default=REPO/'reports')
    a = ap.parse_args()
    torch.set_num_threads(8); torch.set_default_dtype(torch.float64)
    a.out.mkdir(parents=True, exist_ok=True)
    clock = time.time()
    watched = {str(p): sha(p) for p in sorted((a.v2/'first'/'phase1').glob('*.np[yz]'))}
    watched.update({str(REPO/f): sha(REPO/f) for f in LOCAL})
    mods, blobs = load_pinned(a.reference_repo, a.out/'pinned_import')
    mcs, sn_mod = mods['motion_coherence_support'], mods['supernodes']
    from pointwise_rgbd import supported_coherence as sc
    from pointwise_rgbd.image_coherence import build_supernodes as local_build, canonical_knn_graph as local_knn

    v2 = load_run(a.v2)
    K, N = v2['xyz'].shape[:2]
    # ---------------- Stage A: frozen inputs and manifest
    bundle = a.out/'parity_inputs.npz'
    np.savez(bundle, xyz=v2['xyz'], observed=v2['observed'], canonical=v2['canonical'], v2_membership=v2['reps']['membership'],
             phase1_membership=v2['phase1_membership'], times=v2['times'])
    manifest = {'commit': COMMIT, 'pinned_blobs': blobs, 'local_sources_sha256': {f: watched[str(REPO/f)] for f in LOCAL},
                'bundle': {'path': str(bundle), 'sha256': sha(bundle)},
                'inputs': {k: {'path': str(v2['dir']/'phase1'/k), 'sha256': watched.get(str(v2['dir']/'phase1'/k))}
                           for k in ('state.npy', 'point_observed.npy', 'partition.npz', 'covariance.npy')},
                'arrays': {'xyz': [str(v2['xyz'].dtype), list(v2['xyz'].shape)], 'observed': [str(v2['observed'].dtype), list(v2['observed'].shape)],
                           'canonical': [str(v2['canonical'].dtype), list(v2['canonical'].shape)], 'times': [str(v2['times'].dtype), list(v2['times'].shape)]},
                'direction': 'forward; trajectories chronological (frame 0 first); canonical endpoint = frame 0 = partition.xyz',
                'canonical_equals_frame0': bool(np.allclose(v2['canonical'], v2['xyz'][0])),
                'counts': {'individual_6D_point_states': int(N), 'phase1_computational_groups': int(v2['phase1_membership'].max())+1,
                           'phase1_group_size_quantiles': np.quantile(np.bincount(v2['phase1_membership']), [0, .25, .5, .75, 1]).tolist(),
                           'coherence_representative_nodes_v2': int(v2['reps']['membership'].max())+1,
                           'coherence_node_size_quantiles_v2': np.quantile(np.bincount(v2['reps']['membership']), [0, .25, .5, .75, 1]).tolist()},
                'v2_trajectory_policy_fed_to_coherence': 'image-supported node displacements (supported_coherence.supported_node_trajectories: '
                    'members with an active rendered contributor at frame k; endpoint frame always supported) with node-frame weights; '
                    'NOT raw filtered XYZ, NOT zero-order hold',
                'parameters': {'num_supernodes': 192, 'seed': 0, 'knn': 12, 'folds': 5, 'max_candidates': 48, 'min_merge_support': 2, 'min_nodes': 3},
                'invoked_functions': {
                    'v2_production': ['image_coherence.build_supernodes (float32 port of pinned build_supernodes, assignment only)',
                                      'supported_coherence.supported_node_trajectories', 'image_coherence.canonical_knn_graph (port)',
                                      'supported_coherence.weighted_edge_strain (+ removal of gauge-only edges)',
                                      'supported_coherence.extract_moving_component (pinned candidate_hierarchy/edge_distributions/'
                                      'select_moving_component from the vendored copy; weighted Kabsch/CV/complement/refine)'],
                    'pinned_runner': ['supernodes.build_supernodes(mu0.float(), num_supernodes=192, seed=0)', 'motion_coherence_support.run_motion_coherence '
                                      '-> supernode_trajectories (uniform), canonical_knn_graph(traj[0], k=12), edge_strain', 'coherence_extraction.extract_moving_component(folds=5)']}}
    X_pts = torch.as_tensor(v2['xyz']); obs = torch.as_tensor(v2['observed'])

    # ---------------- partitions
    sid_v2 = torch.as_tensor(v2['reps']['membership']).long()
    part_ref = sn_mod.build_supernodes(torch.as_tensor(v2['canonical']).float(), num_supernodes=192, seed=0)
    sid_ref = part_ref.supernode_id.long()
    sid_local_rebuilt = torch.as_tensor(local_build(v2['canonical'], 192)).long()
    manifest['partitions'] = {'v2_saved_vs_pinned_rebuild': match_partitions(sid_v2.numpy(), sid_ref.numpy()),
                              'v2_saved_vs_local_rebuild': match_partitions(sid_v2.numpy(), sid_local_rebuilt.numpy()),
                              'v2_saved_vs_phase1_groups': match_partitions(sid_v2.numpy(), v2['phase1_membership']),
                              'v2_saved_equals_pinned_exactly': bool(torch.equal(sid_v2, sid_ref))}

    records, node_rows, cand_rows = [], [], []
    stage = {}

    def policy(sid):
        sn = mcs.supernode_trajectories([X_pts[k] for k in range(K)], sid)            # pinned, uniform weights
        valid = sn['valid']; Xraw = sn['trajectories'][:, valid, :]
        Xsup, om, ids, _ = sc.supported_node_trajectories(X_pts, obs, sid, 0)
        assert torch.equal(ids, torch.nonzero(valid).reshape(-1)), 'node index spaces differ'
        return {'raw': (Xraw, torch.ones_like(om)), 'supported': (Xsup, om)}, torch.nonzero(valid).reshape(-1)

    for pname, sid in (('B1_v2_partition', sid_v2), ('B2_pinned_partition', sid_ref)):
        pol, node_ids = policy(sid)
        g = mcs.canonical_knn_graph(pol['raw'][0][0], k=12)
        ei, ej = g['edge_i'].long(), g['edge_j'].long()
        n = pol['raw'][0].shape[1]
        gl_i, gl_j = local_knn(pol['raw'][0][0], 12)
        stage[pname] = {'nodes': int(n), 'graph_edges': int(len(ei)), 'graph_components': components(n, ei, ej),
                        'local_graph_equals_pinned': node_pairs(gl_i.numpy(), gl_j.numpy()) == node_pairs(ei.numpy(), ej.numpy()),
                        'node_traj_time0_raw_equals_supported': bool(torch.allclose(pol['raw'][0][0], pol['supported'][0][0], atol=1e-12)),
                        'raw_vs_supported_max_node_difference_m': float((pol['raw'][0]-pol['supported'][0]).norm(dim=2).max())}
        if pname == 'B1_v2_partition':
            saved = v2['reps']
            stage[pname]['saved_v2_edges_equal_pinned_graph'] = node_pairs(saved['edges'][:, 0], saved['edges'][:, 1]) == node_pairs(ei.numpy(), ej.numpy())
            stage[pname]['saved_v2_trajectory_equals_supported_replay'] = bool(np.allclose(saved['trajectory'], pol['supported'][0].numpy(), atol=1e-12))
        for ext in ('pinned', 'local'):
            for pol_name in ('raw', 'supported'):
                X, om = pol[pol_name]
                for refine in (False, True):
                    name = f'{pname}|{ext}_extractor|{pol_name}_trajectories|refine={refine}'
                    r = run_cell(name, X, om, ei, ej, mods, ext, refine)
                    r['partition'] = pname; r['trajectory_policy'] = pol_name
                    r['point_level_moving'] = int(torch.isin(sid, node_ids[torch.as_tensor(r['final_moving_nodes'], dtype=torch.long)]).sum()) if r['final_moving_nodes'] else 0
                    r['node_ids_are_identity'] = bool(torch.equal(node_ids, torch.arange(len(node_ids))))
                    records.append(r)
                    print(f"{name}: {r['status']} seed={len(r['seed_nodes'])} final={len(r['final_moving_nodes'])} points={r['point_level_moving']}", flush=True)
                    for c in r['candidates']: cand_rows.append({'case': name, **{k: v for k, v in c.items()}})
                    if 'refinement' in r:
                        for i in range(len(r['refinement']['E_M'])):
                            node_rows.append({'case': name, 'node': i, 'in_seed': i in set(r['seed_nodes']), 'recruited': i in set(r['refinement']['recruited']),
                                              'final_moving': i in set(r['final_moving_nodes']), 'E_M': r['refinement']['E_M'][i], 'E_I': r['refinement']['E_I'][i],
                                              'E_M_minus_E_I': r['refinement']['E_M'][i]-r['refinement']['E_I'][i], 'label': r['labels'][i]})
    # does the replay reproduce the saved v2 production result?
    prod = next(r for r in records if r['name'] == 'B1_v2_partition|local_extractor|supported_trajectories|refine=True')
    saved_labels = list(v2['reps']['labels'])
    stage['v2_production_replay'] = {'labels_equal_saved': prod['labels'] == saved_labels,
                                     'final_equals_saved': sorted(np.flatnonzero(np.asarray(saved_labels) == 'MOVING_RIGID').tolist()) == prod['final_moving_nodes']}
    # parity of the two extractors on identical raw inputs (local with omega == 1 must equal pinned)
    for pname in ('B1_v2_partition', 'B2_pinned_partition'):
        for refine in (False, True):
            p = next(r for r in records if r['name'] == f'{pname}|pinned_extractor|raw_trajectories|refine={refine}')
            l = next(r for r in records if r['name'] == f'{pname}|local_extractor|raw_trajectories|refine={refine}')
            stage.setdefault('extractor_parity_on_identical_raw_inputs', {})[f'{pname}|refine={refine}'] = {
                'labels_equal': p['labels'] == l['labels'], 'seed_equal': p['seed_nodes'] == l['seed_nodes'], 'final_equal': p['final_moving_nodes'] == l['final_moving_nodes'],
                'pool_ids_equal': [c['candidate_id'] for c in p['candidates']] == [c['candidate_id'] for c in l['candidates']],
                'selected_equal': p['selected'] == l['selected']}
    # ---------------- earlier trajectories through the identical pinned construction
    if a.v1:
        v1 = load_run(a.v1)
        if not np.array_equal(v1['canonical'], v2['canonical']): raise SystemExit('v1 and v2 catalogues differ: not an input-identical replay')
        X1 = torch.as_tensor(v1['xyz'])
        for tag, Xp in (('v1_trajectories', X1), ('v2_trajectories', X_pts)):
            sn = mcs.supernode_trajectories([Xp[k] for k in range(K)], sid_ref)
            Xn = sn['trajectories'][:, sn['valid'], :]
            g = mcs.canonical_knn_graph(Xn[0], k=12)
            for refine in (False, True):
                name = f'REPLAY_pinned_partition_pinned_extractor_raw|{tag}|refine={refine}'
                r = run_cell(name, Xn, torch.ones(Xn.shape[:2]), g['edge_i'].long(), g['edge_j'].long(), mods, 'pinned', refine)
                r['partition'] = 'B2_pinned_partition'; r['trajectory_policy'] = 'raw'; records.append(r)
                print(f"{name}: {r['status']} seed={len(r['seed_nodes'])} final={len(r['final_moving_nodes'])}", flush=True)
                for c in r['candidates']: cand_rows.append({'case': name, **c})
    # ---------------- toy two-rigid-body invariant
    g = torch.Generator().manual_seed(0)
    base = torch.cat([torch.rand(600, 2, generator=g)-.5, torch.zeros(600, 1)], 1); part = torch.cat([torch.rand(200, 2, generator=g)*.3+.1, torch.full((200, 1), .05)], 1)
    Xt = torch.stack([torch.cat([base, part+torch.tensor([.01, .004, 0.])*k]) for k in range(30)])+torch.randn(30, 800, 3, generator=g)*2e-4
    toy_sid = sn_mod.build_supernodes(Xt[0].float(), num_supernodes=40, seed=0).supernode_id.long()
    sn = mcs.supernode_trajectories([Xt[k] for k in range(30)], toy_sid); Xn = sn['trajectories'][:, sn['valid'], :]
    gt = mcs.canonical_knn_graph(Xn[0], k=12)
    toy = {}
    for ext in ('pinned', 'local'):
        r = run_cell(f'toy|{ext}', Xn, torch.ones(Xn.shape[:2]), gt['edge_i'].long(), gt['edge_j'].long(), mods, ext, True)
        moving_nodes = torch.isin(torch.arange(Xn.shape[1]), torch.as_tensor(r['final_moving_nodes'], dtype=torch.long))
        node_is_part = torch.stack([(toy_sid == s)[600:].sum() > (toy_sid == s).sum()/2 for s in torch.nonzero(sn['valid']).reshape(-1)])   # majority of members
        toy[ext] = {'status': r['status'], 'final_nodes': len(r['final_moving_nodes']), 'part_nodes': int(node_is_part.sum()),
                    'exact_part_recovered': bool(torch.equal(moving_nodes, node_is_part))}
    # ---------------- unchanged-artifact proof
    after = {p: sha(p) for p in watched}
    manifest['unchanged_after_audit'] = {p: watched[p] == after[p] for p in watched}
    out = {'manifest': manifest, 'stages': stage, 'toy_two_body': toy, 'runtime_seconds': time.time()-clock,
           'records': [{k: v for k, v in r.items() if k not in ('candidates',)} | {'candidates': r['candidates']} for r in records]}
    a.report_dir.mkdir(exist_ok=True)
    (a.report_dir/'coherence_reference_parity.json').write_text(json.dumps(out, indent=1, default=float))
    with open(a.report_dir/'coherence_reference_parity.csv', 'w', newline='') as f:
        keys = ['row_type', 'case', 'node', 'in_seed', 'recruited', 'final_moving', 'E_M', 'E_I', 'E_M_minus_E_I', 'label']
        ckeys = sorted({k for r in cand_rows for k in r} - {'case'})
        w = csv.DictWriter(f, fieldnames=keys+ckeys, extrasaction='ignore'); w.writeheader()
        for r in node_rows: w.writerow({'row_type': 'node', **r})
        for r in cand_rows: w.writerow({'row_type': 'candidate', **r})
    print(json.dumps({'stages': stage, 'toy_two_body': toy, 'unchanged': all(manifest['unchanged_after_audit'].values())}, indent=1, default=float))


if __name__ == '__main__':
    main()
