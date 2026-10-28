"""Stage E: coherence of ONE completed free pass (F1 or B1), in its own endpoint gauge (contract section 4).

The verified extractor is NOT modified: estimator/motion_coherence_support.run (pinned build_supernodes(192),
canonical kNN(12) graph at the anchor, weighted edge strain, pinned candidate hierarchy / held-out rigidity / seed
selection / refinement; parity: tests/test_coherence_parity.py). It is run twice on the SAME free trajectories:

  pinned_raw            uniform node-frame weights (omega = 1): the pinned reference statistic
  measurement_supported omega from the projective measurement validity of the new filter: a point counts at frame k
                        iff it had >= 1 valid measured RGB-D row at its final linearization (the explicitly named
                        RGB-D adaptation; prediction-only states are not read as observed motion)

`--coherence-weighting` names the policy whose core drives the rigid pass; both results are always written.
Outcome rules (no thresholds added): the pinned status is reported as is; a final core containing EVERY node is
reported UNRESOLVED_FULL_GRAPH_SELECTED (no static reference remains, contract Gate E); an empty core or a
non-RESOLVED pinned status is NO_IDENTIFIABLE_MOVING_CORE. Seed and recruited nodes are kept separately.
"""
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from estimator.motion_coherence_support import run as pinned_coherence_run
from utils.io import json_write, serial

POLICIES = ('pinned_raw', 'measurement_supported')


def run_coherence(free_dir, root, count=192, knn=12, folds=5, driving='pinned_raw'):
    free_dir, root = Path(free_dir), Path(root)
    can = np.load(free_dir/'canonical.npz')
    xyz = np.load(free_dir/'posterior_xyz.npy', mmap_mode='r')
    status = np.load(free_dir/'point_status.npy', mmap_mode='r')
    k0 = int(can['canonical_frame'])
    part = SimpleNamespace(xyz=can['canonical_xyz'], count=None)
    out = {}
    for policy in POLICIES:
        observed = np.ones(status.shape, bool) if policy == 'pinned_raw' else (np.asarray(status) != 2)   # 2 = PREDICTION_ONLY
        res = pinned_coherence_run(part, {'xyz': xyz, 'point_observed': observed}, np.zeros(xyz.shape[0]), root/policy, k0, count, knn, folds)
        n_nodes = int(res['representative_nodes'])
        seed = list(res.get('seed_core_nodes', []))
        recruited = list((res.get('refinement') or {}).get('recruited_nodes', []))
        core = np.flatnonzero(np.asarray(res['point_support'], bool))
        if res['status'] != 'RESOLVED' or len(core) == 0:
            verdict = 'NO_IDENTIFIABLE_MOVING_CORE'
        elif int(res.get('core_nodes', 0)) >= n_nodes:
            verdict = 'UNRESOLVED_FULL_GRAPH_SELECTED'
        else:
            verdict = 'RESOLVED'
        np.save(root/policy/'moving_core_point_ids.npy', core)
        out[policy] = {'pinned_status': res['status'], 'verdict': verdict, 'representative_nodes': n_nodes, 'seed_core_nodes': seed,
                       'recruited_nodes': recruited, 'final_core_nodes': int(res.get('core_nodes', 0)), 'core_points': int(len(core)),
                       'points': int(len(can['canonical_xyz'])), 'labels_counts': res.get('counts'),
                       'node_frame_support': res.get('node_frame_support'), 'core_max_node_displacement_m': res.get('core_max_node_displacement_m'),
                       'moving_core_point_ids_file': str(root/policy/'moving_core_point_ids.npy')}
    a, b = (out[p] for p in POLICIES)
    summary = {'pass_input': str(free_dir), 'canonical_frame': k0, 'driving_policy': driving, 'policies': out,
               'policy_agreement': {'same_verdict': a['verdict'] == b['verdict'], 'same_seed_nodes': sorted(a['seed_core_nodes']) == sorted(b['seed_core_nodes']),
                                    'core_point_jaccard': _jaccard(np.load(root/POLICIES[0]/'moving_core_point_ids.npy'), np.load(root/POLICIES[1]/'moving_core_point_ids.npy'))},
               'status': out[driving]['verdict'], 'moving_core_point_ids_file': out[driving]['moving_core_point_ids_file'],
               'graph': f'pinned build_supernodes({count}) on canonical XYZ, canonical kNN({knn}) at the anchor, {folds}-fold held-out rigidity',
               'ground_truth_used': False}
    json_write(root/'coherence_summary.json', serial(summary))
    return summary


def _jaccard(a, b):
    a, b = set(np.asarray(a).tolist()), set(np.asarray(b).tolist())
    return float(len(a & b)/len(a | b)) if a | b else None
